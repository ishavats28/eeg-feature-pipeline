"""Unified single-recording execution for the EEG feature pipeline."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .bids import Recording, load_recording, select_recording, validate_dataset
from .config import PipelineConfig
from .derivatives import (
    OutputTransaction,
    build_layout,
    derivative_dataset_description,
    expected_recording_paths,
)
from .features import (
    FeatureExtractionError,
    FeatureExtractionResult,
    extract_features,
    save_feature_tables,
)
from .plots import PlotError, generate_diagnostic_figures
from .preprocessing import PreprocessingError, PreprocessingResult, preprocess_raw
from .qc import QCResult, collect_qc_metrics, save_qc_metrics
from .reliability import (
    ReliabilityError,
    StabilityResult,
    analyze_within_recording_stability,
    save_stability_results,
)
from .reporting import generate_html_report
from .sensitivity import (
    SensitivityError,
    SensitivityResult,
    analyze_preprocessing_sensitivity,
    save_sensitivity_results,
)


class PipelineExecutionError(RuntimeError):
    """Raised when unified execution cannot produce a complete output package."""


@dataclass(frozen=True)
class PipelineRunResult:
    """Structured summary of one published recording-level execution."""

    recording: Recording
    derivative_root: Path
    outputs: dict[str, Path]
    stages: dict[str, dict[str, Any]]
    preprocessing: dict[str, Any]
    warnings: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        epochs = self.preprocessing["epochs"]
        return {
            "status": "completed",
            "recording": self.recording.label,
            "derivative_root": str(self.derivative_root),
            "sampling_frequency_hz": self.preprocessing["sampling_frequency_hz"],
            "eeg_channels": self.preprocessing["eeg_channels"],
            "epochs": {
                "initial": epochs["initial"],
                "rejected": epochs["initial"] - epochs["retained"],
                "retained": epochs["retained"],
            },
            "stages": self.stages,
            "outputs": {name: str(path) for name, path in self.outputs.items()},
            "warnings": list(self.warnings),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineExecutionError(f"Could not write '{path.name}': {exc}") from exc


def _software_versions() -> dict[str, str]:
    packages = (
        "eeg-feature-pipeline",
        "mne",
        "mne-bids",
        "mne-connectivity",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "PyYAML",
    )
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    result["eeg-feature-pipeline"] = __version__
    return result


def _configuration_snapshot(config: PipelineConfig, recording: Recording) -> dict[str, Any]:
    effective = asdict(config)
    effective.pop("source_path", None)
    effective["dataset"]["root"] = "<source-bids-root>"
    effective["dataset"]["derivatives_root"] = "<derivative-output-root>"
    return {
        "pipeline_version": __version__,
        "source_configuration_file": config.source_path.name,
        "path_note": "Dataset roots are intentionally represented without private absolute paths.",
        "selected_recording": {
            "path": recording.label,
            "participant": recording.participant,
            "session": recording.session,
            "task": recording.task,
            "acquisition": recording.acquisition,
            "run": recording.run,
            "recording": recording.recording,
        },
        "effective_configuration": _json_value(effective),
    }


def _feature_metadata(config: PipelineConfig) -> dict[str, Any]:
    return {
        "Description": "Column metadata for pipeline-specific EEG feature tables.",
        "FrequencyBandsHz": {
            name: [low, high] for name, (low, high) in config.features.frequency_bands.items()
        },
        "Spectral": {
            "absolute_power_uv2": "Welch PSD integrated by trapezoidal integration; µV².",
            "relative_power": "Band power divided by integrated power over analysis_range_hz.",
            "spectral_entropy": "Normalized base-2 Shannon entropy over PSD bins.",
        },
        "Temporal": {
            "rms_uv": "Root mean square amplitude; µV.",
            "variance_uv2": "Population variance (ddof=0); µV².",
            "hjorth_activity_uv2": "Population variance; µV².",
            "hjorth_mobility": "Square root of first-difference variance / signal variance.",
            "hjorth_complexity": "First-difference mobility / signal mobility.",
        },
        "Connectivity": {
            "AnalysisLevel": "recording_pooled_epochs",
            "Interpretation": (
                "Sensor-space statistical dependence; not anatomical or source-level connectivity."
            ),
        },
    }


def _epochs_sidecar(
    config: PipelineConfig, recording: Recording, preprocessing: PreprocessingResult
) -> dict[str, Any]:
    return {
        "Description": "Pipeline-specific MNE fixed-length cleaned Epochs object.",
        "SourceRecording": recording.label,
        "SamplingFrequency": preprocessing.final_sampling_frequency_hz,
        "EEGChannelCount": preprocessing.retained_eeg_channel_count,
        "EpochDurationSeconds": config.epochs.duration_seconds,
        "EpochOverlapSeconds": config.epochs.overlap_seconds,
        "RetainedEpochCount": preprocessing.retained_epoch_count,
        "Format": "MNE FIF",
        "BIDSStandardizedDataType": False,
    }


def _relative(final_path: Path, root: Path) -> str:
    return final_path.relative_to(root).as_posix()


def run_pipeline(
    config: PipelineConfig,
    *,
    participant: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> PipelineRunResult:
    """Run all configured stages for exactly one selected BIDS EEG recording."""
    selectors = {
        "participant": participant if participant is not None else config.dataset.participant,
        "session": session if session is not None else config.dataset.session,
        "task": task if task is not None else config.dataset.task,
        "run": run if run is not None else config.dataset.run,
    }
    recording = select_recording(config.dataset.root, **selectors)
    source_description = validate_dataset(config.dataset.root)
    layout = build_layout(
        output_dir if output_dir is not None else config.dataset.derivatives_root,
        recording,
        exact_root=output_dir is not None,
    )
    allow_overwrite = overwrite or config.reporting.overwrite
    transaction = OutputTransaction(
        layout,
        expected_recording_paths(layout, config),
        overwrite=allow_overwrite,
    )
    stages: dict[str, dict[str, Any]] = {
        "selection": {"status": "completed", "reason": None},
        "preprocessing": {"status": "pending", "reason": None},
        "features": {"status": "pending", "reason": None},
        "connectivity": {
            "status": "pending" if config.connectivity.enabled else "skipped",
            "reason": None if config.connectivity.enabled else "disabled by configuration",
        },
        "stability": {
            "status": "pending" if config.reliability.enabled else "skipped",
            "reason": None if config.reliability.enabled else "disabled by configuration",
        },
        "sensitivity": {
            "status": "pending" if config.sensitivity.enabled else "skipped",
            "reason": None if config.sensitivity.enabled else "disabled by configuration",
        },
        "qc": {
            "status": "pending" if config.reporting.qc_enabled else "skipped",
            "reason": None if config.reporting.qc_enabled else "disabled by configuration",
        },
        "figures": {
            "status": "pending" if config.reporting.save_figures else "skipped",
            "reason": None if config.reporting.save_figures else "disabled by configuration",
        },
        "report": {"status": "pending", "reason": None},
    }
    warnings: list[str] = []
    outputs: dict[str, Path] = {"dataset_description": layout.root / "dataset_description.json"}

    with transaction as staging:
        source_raw = load_recording(recording)
        preprocessing = preprocess_raw(source_raw, config)
        if preprocessing.retained_epoch_count == 0:
            raise PipelineExecutionError("Preprocessing retained zero epochs.")
        stages["preprocessing"]["status"] = "completed"
        warnings.extend(preprocessing.warnings)

        features: FeatureExtractionResult = extract_features(
            preprocessing.epochs, config, recording
        )
        stages["features"]["status"] = "completed"
        warnings.extend(features.diagnostics["connectivity"]["warnings"])
        if config.connectivity.enabled:
            stages["connectivity"]["status"] = "completed"
            stages["connectivity"]["rows"] = len(features.connectivity_table)
        feature_paths = save_feature_tables(features, recording, staging.recording_dir)
        for name, staged_path in feature_paths.items():
            outputs[f"{name}_features"] = layout.recording_dir / staged_path.relative_to(
                staging.recording_dir
            )

        stability: StabilityResult | None = None
        if config.reliability.enabled:
            try:
                stability = analyze_within_recording_stability(
                    features, config.reliability, config.epochs
                )
            except ReliabilityError as exc:
                stages["stability"] = {"status": "unavailable", "reason": str(exc)}
                warnings.append(f"Stability unavailable: {exc}")
            if stability is not None:
                stages["stability"]["status"] = stability.summary["status"]
                stages["stability"]["matched_features"] = stability.summary.get(
                    "matched_feature_count", 0
                )
                stability_paths = save_stability_results(
                    stability, recording, staging.recording_dir
                )
                for name, staged_path in stability_paths.items():
                    outputs[f"stability_{name}"] = layout.recording_dir / staged_path.relative_to(
                        staging.recording_dir
                    )
                warnings.extend(stability.summary["warnings"])

        sensitivity: SensitivityResult | None = None
        if config.sensitivity.enabled:
            try:
                sensitivity = analyze_preprocessing_sensitivity(
                    source_raw,
                    config,
                    recording,
                    baseline_result=preprocessing,
                    baseline_features=features,
                )
            except (SensitivityError, PreprocessingError, FeatureExtractionError) as exc:
                stages["sensitivity"] = {"status": "unavailable", "reason": str(exc)}
                warnings.append(f"Sensitivity unavailable: {exc}")
            if sensitivity is not None:
                stages["sensitivity"]["status"] = sensitivity.comparison["status"]
                stages["sensitivity"]["matched_feature_rows"] = sensitivity.comparison[
                    "matched_feature_rows"
                ]
                sensitivity_paths = save_sensitivity_results(
                    sensitivity, recording, staging.recording_dir
                )
                for name, staged_path in sensitivity_paths.items():
                    outputs[f"sensitivity_{name}"] = layout.recording_dir / staged_path.relative_to(
                        staging.recording_dir
                    )
                warnings.extend(sensitivity.comparison["warnings"])

        qc: QCResult | None = None
        if config.reporting.qc_enabled:
            qc = collect_qc_metrics(preprocessing)
            qc_path = save_qc_metrics(qc, recording, staging.recording_dir)
            outputs["qc_metrics"] = layout.recording_dir / qc_path.relative_to(
                staging.recording_dir
            )
            stages["qc"]["status"] = "completed"

        figure_paths: dict[str, Path] = {}
        if config.reporting.save_figures:
            try:
                plot_result = generate_diagnostic_figures(
                    source_raw,
                    preprocessing,
                    recording,
                    staging.figures_dir,
                    features=features,
                    stability=stability,
                    sensitivity=sensitivity,
                    dpi=config.reporting.figure_dpi,
                )
                figure_paths = plot_result.paths
                stages["figures"]["status"] = "completed"
                stages["figures"]["count"] = len(figure_paths)
                warnings.extend(plot_result.warnings)
                for name, staged_path in figure_paths.items():
                    outputs[f"figure_{name}"] = layout.recording_dir / staged_path.relative_to(
                        staging.recording_dir
                    )
            except PlotError as exc:
                stages["figures"] = {"status": "failed", "reason": str(exc)}
                raise PipelineExecutionError(
                    f"Requested diagnostic figure generation failed: {exc}"
                ) from exc

        try:
            preprocessing.epochs.save(staging.epochs, overwrite=False, verbose="ERROR")
        except (OSError, RuntimeError, ValueError) as exc:
            raise PipelineExecutionError(f"Could not save cleaned Epochs FIF: {exc}") from exc
        outputs["cleaned_epochs"] = layout.epochs
        _write_json(staging.epochs_sidecar, _epochs_sidecar(config, recording, preprocessing))
        outputs["cleaned_epochs_sidecar"] = layout.epochs_sidecar
        _write_json(staging.feature_metadata, _feature_metadata(config))
        outputs["feature_metadata"] = layout.feature_metadata
        snapshot = _configuration_snapshot(config, recording)
        _write_json(staging.processing_config, snapshot)
        outputs["processing_config"] = layout.processing_config

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        software_versions = _software_versions()
        provenance = {
            "pipeline_version": __version__,
            "processing_timestamp": timestamp,
            "software_versions": software_versions,
            "configuration_snapshot": _relative(layout.processing_config, layout.root),
            "random_seeds": {"ica": config.artifacts.ica_random_state},
            "warnings": list(dict.fromkeys(warnings)),
        }
        outputs["execution_manifest"] = layout.manifest
        report_references = {
            name: _relative(path, layout.root)
            for name, path in {**outputs, "report": layout.report}.items()
        }
        generate_html_report(
            staging.report,
            recording=recording,
            config=config,
            preprocessing=preprocessing,
            features=features,
            qc=qc,
            stability=stability,
            sensitivity=sensitivity,
            figures=figure_paths,
            stages=stages,
            output_references=report_references,
            provenance=provenance,
        )
        outputs["report"] = layout.report
        stages["report"]["status"] = "completed"

        manifest = {
            "status": "completed",
            "pipeline": {
                "name": "eeg-feature-pipeline",
                "version": __version__,
                "author": "Isha Vats",
            },
            "processing_timestamp": timestamp,
            "input_recording": recording.label,
            "selected_entities": snapshot["selected_recording"],
            "enabled_stages": {
                "connectivity": config.connectivity.enabled,
                "stability": config.reliability.enabled,
                "sensitivity": config.sensitivity.enabled,
                "qc": config.reporting.qc_enabled,
                "figures": config.reporting.save_figures,
            },
            "stages": stages,
            "observed_preprocessing": preprocessing.summary(),
            "configuration_snapshot": _relative(layout.processing_config, layout.root),
            "software_versions": software_versions,
            "random_seeds": {"ica": config.artifacts.ica_random_state},
            "outputs": {name: _relative(path, layout.root) for name, path in outputs.items()},
            "warnings": list(dict.fromkeys(warnings)),
            "derivative_compatibility": {
                "dataset_description": "BIDS derivative root metadata",
                "analysis_files": "pipeline-specific; not blanket BIDS Derivatives compliance",
            },
        }
        _write_json(staging.manifest, manifest)
        transaction.commit(derivative_dataset_description(source_description))

    return PipelineRunResult(
        recording=recording,
        derivative_root=layout.root,
        outputs=outputs,
        stages=stages,
        preprocessing=preprocessing.summary(),
        warnings=tuple(dict.fromkeys(warnings)),
    )
