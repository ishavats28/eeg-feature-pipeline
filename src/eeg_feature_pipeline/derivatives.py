"""Derivative-style paths and safe staged output publication."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import PipelineConfig

PIPELINE_NAME = "eeg-feature-pipeline"


class DerivativeError(RuntimeError):
    """Raised when derivative outputs cannot be organized or published safely."""


def entity_stem(recording: Any) -> str:
    """Return source entities in BIDS canonical order where applicable."""
    entities = [f"sub-{recording.participant}"]
    for attribute, label in (
        ("session", "ses"),
        ("task", "task"),
        ("acquisition", "acq"),
        ("run", "run"),
        ("recording", "recording"),
    ):
        value = getattr(recording, attribute, None)
        if value:
            entities.append(f"{label}-{value}")
    return "_".join(entities)


def resolve_pipeline_root(path: str | Path, *, exact: bool = False) -> Path:
    """Resolve a configured derivatives parent or an explicit pipeline root."""
    candidate = Path(path).expanduser().resolve()
    if exact or candidate.name == PIPELINE_NAME:
        return candidate
    return candidate / PIPELINE_NAME


@dataclass(frozen=True)
class DerivativeLayout:
    """Deterministic locations for one recording's output package."""

    root: Path
    recording_dir: Path
    figures_dir: Path
    stem: str

    def path(self, descriptor: str, suffix: str, extension: str) -> Path:
        return self.recording_dir / f"{self.stem}_desc-{descriptor}_{suffix}.{extension}"

    @property
    def epochs(self) -> Path:
        return self.path("cleaned", "epo", "fif")

    @property
    def epochs_sidecar(self) -> Path:
        return self.path("cleaned", "epo", "json")

    @property
    def spectral(self) -> Path:
        return self.path("spectral", "features", "tsv")

    @property
    def temporal(self) -> Path:
        return self.path("temporal", "features", "tsv")

    @property
    def connectivity(self) -> Path:
        return self.path("connectivity", "features", "tsv")

    @property
    def feature_metadata(self) -> Path:
        return self.path("features", "metadata", "json")

    @property
    def stability_features(self) -> Path:
        return self.path("stability", "features", "tsv")

    @property
    def stability_summary(self) -> Path:
        return self.path("stability", "summary", "json")

    @property
    def sensitivity_features(self) -> Path:
        return self.path("sensitivity", "features", "tsv")

    @property
    def sensitivity_config(self) -> Path:
        return self.path("sensitivity", "config", "json")

    @property
    def qc_metrics(self) -> Path:
        return self.path("qc", "metrics", "json")

    @property
    def processing_config(self) -> Path:
        return self.path("processing", "config", "json")

    @property
    def manifest(self) -> Path:
        return self.path("execution", "manifest", "json")

    @property
    def report(self) -> Path:
        return self.recording_dir / f"{self.stem}_desc-report.html"


def build_layout(
    derivatives_root: str | Path,
    recording: Any,
    *,
    exact_root: bool = False,
) -> DerivativeLayout:
    root = resolve_pipeline_root(derivatives_root, exact=exact_root)
    subject = root / f"sub-{recording.participant}"
    if recording.session:
        subject /= f"ses-{recording.session}"
    recording_dir = subject / "eeg"
    return DerivativeLayout(root, recording_dir, recording_dir / "figures", entity_stem(recording))


def layout_in_directory(directory: Path, stem: str) -> DerivativeLayout:
    """Create a layout rooted directly at a staging recording directory."""
    return DerivativeLayout(directory.parent, directory, directory / "figures", stem)


def expected_recording_paths(layout: DerivativeLayout, config: PipelineConfig) -> tuple[Path, ...]:
    """Return a conservative superset of files a configured run may publish."""
    paths = [
        layout.epochs,
        layout.epochs_sidecar,
        layout.spectral,
        layout.temporal,
        layout.feature_metadata,
        layout.processing_config,
        layout.manifest,
        layout.report,
    ]
    if config.connectivity.enabled:
        paths.append(layout.connectivity)
    if config.reliability.enabled:
        paths.extend((layout.stability_summary, layout.stability_features))
    if config.sensitivity.enabled:
        paths.extend((layout.sensitivity_features, layout.sensitivity_config))
    if config.reporting.qc_enabled:
        paths.append(layout.qc_metrics)
    if config.reporting.save_figures:
        for descriptor in (
            "qc_signal",
            "qc_psd",
            "qc_band-power",
            "qc_epoch-retention",
            "stability_comparison",
            "sensitivity_comparison",
        ):
            paths.append(layout.figures_dir / f"{layout.stem}_desc-{descriptor}.png")
    return tuple(paths)


def derivative_dataset_description(source: dict[str, Any]) -> dict[str, Any]:
    """Build required derivative dataset metadata without invented source identifiers."""
    return {
        "Name": "EEG Feature Pipeline derivatives",
        "BIDSVersion": str(source["BIDSVersion"]),
        "DatasetType": "derivative",
        "Authors": ["Isha Vats"],
        "GeneratedBy": [
            {
                "Name": PIPELINE_NAME,
                "Version": __version__,
                "Description": (
                    "Resting-state EEG preprocessing, feature extraction, stability, "
                    "sensitivity, and quality-control outputs."
                ),
            }
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class OutputTransaction:
    """Stage one recording and publish known files with the manifest last."""

    def __init__(
        self,
        layout: DerivativeLayout,
        expected_paths: tuple[Path, ...],
        *,
        overwrite: bool,
    ) -> None:
        self.layout = layout
        self.expected_paths = expected_paths
        self.overwrite = overwrite
        self.staging_root: Path | None = None
        self.staging_layout: DerivativeLayout | None = None

    def check_conflicts(self) -> None:
        conflicts = sorted(path for path in self.expected_paths if path.exists())
        if conflicts and not self.overwrite:
            raise DerivativeError(
                "Output already exists and overwrite is disabled: "
                + ", ".join(str(path) for path in conflicts[:8])
            )

    def check_dataset_metadata(self) -> None:
        dataset_path = self.layout.root / "dataset_description.json"
        if not dataset_path.exists():
            return
        try:
            existing = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DerivativeError(
                f"Existing derivative dataset description is invalid: {exc}"
            ) from exc
        generated = existing.get("GeneratedBy") if isinstance(existing, dict) else None
        if (
            not isinstance(existing, dict)
            or existing.get("DatasetType") != "derivative"
            or not isinstance(generated, list)
            or not generated
            or not isinstance(generated[0], dict)
            or generated[0].get("Name") != PIPELINE_NAME
        ):
            raise DerivativeError(
                "Output root contains dataset metadata from a different derivative pipeline."
            )

    def __enter__(self) -> DerivativeLayout:
        self.check_conflicts()
        self.check_dataset_metadata()
        self.layout.root.mkdir(parents=True, exist_ok=True)
        created = tempfile.mkdtemp(prefix=f".{self.layout.stem}.staging-", dir=self.layout.root)
        self.staging_root = Path(created).resolve()
        recording_dir = self.staging_root / "recording"
        recording_dir.mkdir(parents=True)
        self.staging_layout = layout_in_directory(recording_dir, self.layout.stem)
        return self.staging_layout

    def _validate_staging(self) -> Path:
        if self.staging_root is None or self.staging_layout is None:
            raise DerivativeError("Output transaction was not started.")
        root = self.layout.root.resolve()
        staging = self.staging_root.resolve()
        if staging.parent != root or not staging.name.startswith(f".{self.layout.stem}.staging-"):
            raise DerivativeError("Refusing to operate on an unexpected staging path.")
        return staging

    def commit(self, dataset_description: dict[str, Any]) -> dict[str, Path]:
        self._validate_staging()
        assert self.staging_layout is not None
        manifest_relative = self.staging_layout.manifest.relative_to(
            self.staging_layout.recording_dir
        )
        if not self.staging_layout.manifest.is_file():
            raise DerivativeError(
                "Staged execution manifest is missing; outputs were not published."
            )
        self.check_conflicts()
        self.check_dataset_metadata()
        dataset_path = self.layout.root / "dataset_description.json"
        if not dataset_path.exists():
            assert self.staging_root is not None
            temporary_description = self.staging_root / "dataset_description.json"
            _write_json(temporary_description, dataset_description)
            temporary_description.replace(dataset_path)

        staged_files = sorted(
            (path for path in self.staging_layout.recording_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(self.staging_layout.recording_dir).as_posix(),
        )
        non_manifest = [
            path
            for path in staged_files
            if path.relative_to(self.staging_layout.recording_dir) != manifest_relative
        ]
        if self.overwrite and self.layout.manifest.exists():
            self.layout.manifest.unlink()
        published: dict[str, Path] = {}
        try:
            for source in [*non_manifest, self.staging_layout.manifest]:
                relative = source.relative_to(self.staging_layout.recording_dir)
                target = self.layout.recording_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                published[relative.as_posix()] = target
        except OSError as exc:
            raise DerivativeError(
                "Could not publish the complete output package; no new completion "
                "manifest was left. "
                f"Details: {exc}"
            ) from exc
        self._cleanup()
        return published

    def _cleanup(self) -> None:
        if self.staging_root is None or not self.staging_root.exists():
            return
        staging = self._validate_staging()
        shutil.rmtree(staging)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._cleanup()
