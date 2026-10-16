"""Read-only discovery and inspection of BIDS EEG recordings."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mne.io import BaseRaw


class BIDSError(RuntimeError):
    """Base class for user-facing BIDS ingestion failures."""


class BIDSDependencyError(BIDSError):
    """Raised when MNE-BIDS is unavailable."""


class BIDSSelectionError(BIDSError):
    """Raised when recording selectors match zero or multiple recordings."""


_EEG_DATA_EXTENSIONS = (".vhdr", ".edf", ".bdf", ".set")


@dataclass(frozen=True)
class Recording:
    """A lightweight reference to one BIDS EEG data file."""

    path: Any
    relative_path: Path
    participant: str
    session: str | None
    task: str | None
    run: str | None
    acquisition: str | None
    recording: str | None
    extension: str | None

    @property
    def label(self) -> str:
        return self.relative_path.as_posix()


def _mne_bids() -> Any:
    try:
        import mne_bids
    except ImportError as exc:
        raise BIDSDependencyError(
            "MNE-BIDS is not installed. Install the project dependencies before reading BIDS data."
        ) from exc
    return mne_bids


def validate_dataset(root: str | Path) -> dict[str, Any]:
    """Validate required root-level BIDS dataset metadata without modifying it."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise BIDSError(f"BIDS dataset root does not exist or is not a directory: {root}")
    description_path = root / "dataset_description.json"
    if not description_path.is_file():
        raise BIDSError(f"BIDS dataset is missing required file: {description_path}")
    try:
        description = json.loads(description_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BIDSError(f"Could not read valid JSON from {description_path}: {exc}") from exc
    if not isinstance(description, dict):
        raise BIDSError("BIDS dataset_description.json must contain a JSON object.")
    missing = [key for key in ("Name", "BIDSVersion") if not description.get(key)]
    if missing:
        raise BIDSError(
            "BIDS dataset_description.json is missing required field(s): " + ", ".join(missing)
        )
    return description


def discover_recordings(
    root: str | Path,
    *,
    participant: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
) -> list[Recording]:
    """Discover matching EEG recordings without loading signal samples."""
    root = Path(root).resolve()
    validate_dataset(root)
    mne_bids = _mne_bids()
    query = {
        "root": root,
        "subjects": participant,
        "sessions": session,
        "tasks": task,
        "runs": run,
        "suffixes": "eeg",
        "extensions": _EEG_DATA_EXTENSIONS,
        "datatypes": "eeg",
        "ignore_json": True,
    }
    try:
        paths = mne_bids.find_matching_paths(
            **query,
            check=True,
        )
        if not paths:
            unchecked = mne_bids.find_matching_paths(**query, check=False)
            malformed = [path for path in unchecked if not path.check]
            if malformed:
                preview = "; ".join(str(path.fpath) for path in malformed[:3])
                raise BIDSError(
                    "Found EEG data with non-conforming BIDS paths. "
                    f"Run the BIDS Validator for details. Examples: {preview}"
                )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BIDSError(f"MNE-BIDS could not inspect the dataset: {exc}") from exc
    records = [
        Recording(
            path=path,
            relative_path=Path(path.fpath).relative_to(root),
            participant=path.subject,
            session=path.session,
            task=path.task,
            run=path.run,
            acquisition=path.acquisition,
            recording=path.recording,
            extension=path.extension,
        )
        for path in paths
    ]
    return sorted(records, key=lambda item: item.label)


def select_recording(
    root: str | Path,
    *,
    participant: str | None = None,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
) -> Recording:
    """Require selectors to identify exactly one EEG recording."""
    records = discover_recordings(
        root, participant=participant, session=session, task=task, run=run
    )
    selector = (
        ", ".join(
            f"{key}={value}"
            for key, value in {
                "participant": participant,
                "session": session,
                "task": task,
                "run": run,
            }.items()
            if value is not None
        )
        or "no selectors"
    )
    if not records:
        raise BIDSSelectionError(f"No BIDS EEG recording matched {selector}.")
    if len(records) > 1:
        preview = "; ".join(record.label for record in records[:5])
        extra = f"; and {len(records) - 5} more" if len(records) > 5 else ""
        raise BIDSSelectionError(
            f"Selection is ambiguous: {len(records)} EEG recordings matched {selector}. "
            f"Refine participant/session/task/run. Matches: {preview}{extra}"
        )
    return records[0]


def load_recording(recording: Recording) -> BaseRaw:
    """Load a selected recording lazily through MNE-BIDS."""
    mne_bids = _mne_bids()
    try:
        return mne_bids.read_raw_bids(
            bids_path=recording.path, extra_params={"preload": False}, verbose="ERROR"
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BIDSError(f"Could not load BIDS EEG recording '{recording.label}': {exc}") from exc


def inspect_recording(recording: Recording) -> dict[str, Any]:
    """Read a recording and return serializable acquisition metadata."""
    raw = load_recording(recording)
    info = raw.info
    duration = raw.n_times / float(info["sfreq"])
    line_frequency = info.get("line_freq")
    return {
        "path": recording.label,
        "participant": recording.participant,
        "session": recording.session,
        "task": recording.task,
        "run": recording.run,
        "acquisition": recording.acquisition,
        "format": recording.extension,
        "sampling_frequency_hz": float(info["sfreq"]),
        "duration_seconds": duration,
        "n_samples": int(raw.n_times),
        "n_channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "channel_types": dict(sorted(Counter(raw.get_channel_types()).items())),
        "bad_channels": list(info["bads"]),
        "line_frequency_hz": None if line_frequency is None else float(line_frequency),
        "acquisition_highpass_hz": float(info["highpass"]),
        "acquisition_lowpass_hz": float(info["lowpass"]),
        "annotation_count": len(raw.annotations),
        "annotations": sorted(set(str(value) for value in raw.annotations.description)),
    }
