"""Fixed-length resting-state EEG epoching and rejection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mne

from .config import ArtifactConfig, EpochConfig

if TYPE_CHECKING:
    from mne import Epochs
    from mne.io import BaseRaw


class EpochingError(RuntimeError):
    """Raised when fixed-length resting-state epochs cannot be constructed."""


@dataclass(frozen=True)
class EpochingResult:
    """Epochs and transparent rejection counts."""

    epochs: Epochs
    initial_count: int
    retained_count: int
    rejection_reasons: dict[str, int]
    warnings: tuple[str, ...] = ()


def create_fixed_length_epochs(
    raw: BaseRaw,
    epoch_config: EpochConfig,
    artifact_config: ArtifactConfig,
) -> EpochingResult:
    """Create overlapping resting-state epochs with no baseline correction.

    MNE stores EEG in volts. ``amplitude_threshold_uv`` is converted to volts and
    used as a peak-to-peak rejection threshold when epoch rejection is enabled.
    """
    sfreq = float(raw.info["sfreq"])
    epoch_samples = int(round(epoch_config.duration_seconds * sfreq))
    if epoch_samples < 2:
        raise EpochingError("Epoch duration corresponds to fewer than two samples.")
    if raw.n_times < epoch_samples:
        duration = raw.n_times / sfreq
        raise EpochingError(
            f"Recording is too short for one {epoch_config.duration_seconds:g}-second epoch "
            f"({duration:.3f} seconds available)."
        )
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    if len(picks) == 0:
        raise EpochingError("No usable EEG channels remain for epoching.")
    try:
        events = mne.make_fixed_length_events(
            raw,
            id=1,
            duration=epoch_config.duration_seconds,
            overlap=epoch_config.overlap_seconds,
        )
    except (RuntimeError, ValueError) as exc:
        raise EpochingError(f"Could not construct fixed-length events: {exc}") from exc
    if len(events) == 0:
        raise EpochingError("No complete fixed-length epochs fit within the recording.")
    reject = None
    if artifact_config.reject_bad_epochs:
        reject = {"eeg": artifact_config.amplitude_threshold_uv * 1e-6}
    tmax = (epoch_samples - 1) / sfreq
    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"rest": 1},
            tmin=0.0,
            tmax=tmax,
            baseline=None,
            picks=picks,
            preload=True,
            reject=reject,
            reject_by_annotation=epoch_config.reject_by_annotation,
            verbose="ERROR",
        )
    except (RuntimeError, ValueError) as exc:
        raise EpochingError(f"Could not create fixed-length epochs: {exc}") from exc
    reasons: Counter[str] = Counter()
    for entry in epochs.drop_log:
        for reason in set(entry):
            if reason != "IGNORED":
                reasons[str(reason)] += 1
    retained = len(epochs)
    result_warnings: tuple[str, ...] = ()
    if retained == 0:
        result_warnings = ("All fixed-length epochs were rejected.",)
    return EpochingResult(
        epochs=epochs,
        initial_count=len(events),
        retained_count=retained,
        rejection_reasons=dict(sorted(reasons.items())),
        warnings=result_warnings,
    )
