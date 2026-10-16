"""Resting-state EEG preprocessing with structured diagnostics.

Processing order is channel selection, bad-channel inspection, notch and band-pass
filtering, rereferencing, optional ICA, optional resampling, and fixed-length
epoching. The caller's Raw object is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import numpy as np

from .artifacts import ICAResult, apply_ica_artifact_correction
from .config import ArtifactConfig, PipelineConfig, PreprocessingConfig
from .epoching import EpochingError, EpochingResult, create_fixed_length_epochs

if TYPE_CHECKING:
    from mne import Epochs
    from mne.io import BaseRaw

    from .bids import Recording


class PreprocessingError(RuntimeError):
    """Raised when EEG preprocessing cannot produce a valid processing signal."""


@dataclass(frozen=True)
class BadChannelResult:
    """Existing and automatically detected channel-quality findings."""

    existing: tuple[str, ...]
    detected: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessingResult:
    """Processed data and compact, JSON-friendly diagnostics."""

    raw: BaseRaw
    epochs: Epochs
    original_sampling_frequency_hz: float
    final_sampling_frequency_hz: float
    original_eeg_channel_count: int
    retained_eeg_channel_count: int
    existing_bad_channels: tuple[str, ...]
    detected_bad_channels: tuple[str, ...]
    bad_channel_reasons: dict[str, tuple[str, ...]]
    ica_status: str
    ica_candidate_components: tuple[int, ...]
    ica_removed_components: tuple[int, ...]
    ica_evidence: dict[str, Any] | None
    initial_epoch_count: int
    retained_epoch_count: int
    epoch_rejection_reasons: dict[str, int]
    parameters: dict[str, Any]
    warnings: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        """Return diagnostics without embedding MNE objects."""
        return {
            "sampling_frequency_hz": {
                "original": self.original_sampling_frequency_hz,
                "final": self.final_sampling_frequency_hz,
            },
            "eeg_channels": {
                "original": self.original_eeg_channel_count,
                "retained": self.retained_eeg_channel_count,
            },
            "bad_channels": {
                "existing": list(self.existing_bad_channels),
                "detected": list(self.detected_bad_channels),
                "reasons": {
                    name: list(reasons) for name, reasons in self.bad_channel_reasons.items()
                },
            },
            "ica": {
                "status": self.ica_status,
                "candidate_components": list(self.ica_candidate_components),
                "removed_components": list(self.ica_removed_components),
                "evidence": self.ica_evidence,
            },
            "epochs": {
                "initial": self.initial_epoch_count,
                "retained": self.retained_epoch_count,
                "rejection_reasons": self.epoch_rejection_reasons,
            },
            "parameters": self.parameters,
            "warnings": list(self.warnings),
        }


def select_eeg_and_eog(raw: BaseRaw, eeg_channels: tuple[str, ...] | None = None) -> BaseRaw:
    """Return selected EEG plus any EOG channels used for artifact detection."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    if len(eeg_picks) == 0:
        raise PreprocessingError("The recording contains no EEG channels.")
    available_eeg = {raw.ch_names[index] for index in eeg_picks}
    if eeg_channels is None:
        selected_eeg = [raw.ch_names[index] for index in eeg_picks]
    else:
        missing = [name for name in eeg_channels if name not in raw.ch_names]
        wrong_type = [
            name for name in eeg_channels if name in raw.ch_names and name not in available_eeg
        ]
        if missing:
            raise PreprocessingError(
                "Configured EEG channel(s) were not found: " + ", ".join(missing)
            )
        if wrong_type:
            raise PreprocessingError(
                "Configured analysis channel(s) are not typed as EEG: " + ", ".join(wrong_type)
            )
        selected_eeg = list(eeg_channels)
    eog_picks = mne.pick_types(raw.info, eeg=False, eog=True, exclude=[])
    selected = [*selected_eeg, *(raw.ch_names[index] for index in eog_picks)]
    return raw.copy().pick(selected)


def validate_recording_settings(raw: BaseRaw, config: PreprocessingConfig) -> None:
    """Validate configured frequencies against the recording sampling frequency."""
    sfreq = float(raw.info["sfreq"])
    nyquist = sfreq / 2.0
    if config.highpass_hz >= nyquist:
        raise PreprocessingError(
            f"High-pass frequency {config.highpass_hz:g} Hz must be below the "
            f"recording Nyquist frequency ({nyquist:g} Hz)."
        )
    if config.lowpass_hz >= nyquist:
        raise PreprocessingError(
            f"Low-pass frequency {config.lowpass_hz:g} Hz must be below the "
            f"recording Nyquist frequency ({nyquist:g} Hz)."
        )
    active_notches = [
        frequency
        for frequency in config.notch_hz
        if config.highpass_hz < frequency < config.lowpass_hz
    ]
    invalid_notches = [frequency for frequency in active_notches if frequency >= nyquist]
    if invalid_notches:
        raise PreprocessingError(
            "Notch frequencies must be below the recording Nyquist frequency: "
            + ", ".join(f"{value:g}" for value in invalid_notches)
        )
    if config.resample_hz is not None:
        if config.resample_hz > sfreq:
            raise PreprocessingError(
                f"Configured resampling frequency {config.resample_hz:g} Hz exceeds the "
                f"recording sampling frequency ({sfreq:g} Hz); upsampling is not supported."
            )
        if config.resample_hz / 2.0 <= config.lowpass_hz:
            raise PreprocessingError(
                "The resampled Nyquist frequency must exceed the configured low-pass frequency."
            )


def detect_bad_channels(raw: BaseRaw, config: ArtifactConfig) -> BadChannelResult:
    """Mark flat and robustly unusual high-frequency-noise EEG channels.

    Noisy-channel scoring uses each channel's robust first-difference scale, which
    reduces sensitivity to differing slow physiological amplitudes. Scores are then
    compared across at least four usable EEG channels using a median/MAD z-score.
    """
    existing = tuple(name for name in raw.info["bads"] if name in raw.ch_names)
    if not config.reject_bad_channels:
        return BadChannelResult(existing=existing, detected=(), reasons={})
    eeg_names = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind == "eeg" and name not in existing
    ]
    if not eeg_names:
        return BadChannelResult(
            existing=existing,
            detected=(),
            reasons={},
            warnings=("No EEG channels were available for automatic bad-channel detection.",),
        )
    data = raw.get_data(picks=eeg_names, reject_by_annotation="omit")
    if data.shape[1] < 2:
        return BadChannelResult(
            existing=existing,
            detected=(),
            reasons={},
            warnings=("Too few unannotated samples for automatic bad-channel detection.",),
        )
    reasons: dict[str, list[str]] = {}
    peak_to_peak_uv = np.ptp(data, axis=1) * 1e6
    for name, value in zip(eeg_names, peak_to_peak_uv, strict=True):
        if value <= config.flat_threshold_uv:
            reasons.setdefault(name, []).append(
                f"flat: peak-to-peak {value:.6g} uV <= {config.flat_threshold_uv:g} uV"
            )

    nonflat_indices = [index for index, name in enumerate(eeg_names) if name not in reasons]
    quality_warnings: list[str] = []
    if len(nonflat_indices) >= 4:
        differences = np.diff(data[nonflat_indices], axis=1)
        centers = np.median(differences, axis=1, keepdims=True)
        noise_scales = 1.4826 * np.median(np.abs(differences - centers), axis=1)
        positive = noise_scales > np.finfo(float).tiny
        if np.count_nonzero(positive) >= 4:
            log_scales = np.log(noise_scales[positive])
            median = float(np.median(log_scales))
            mad = float(np.median(np.abs(log_scales - median)))
            if mad > np.finfo(float).eps:
                robust_z = 0.67448975 * (log_scales - median) / mad
                valid_indices = np.asarray(nonflat_indices)[positive]
                for index, score in zip(valid_indices, robust_z, strict=True):
                    if score > config.noisy_channel_zscore:
                        name = eeg_names[int(index)]
                        reasons.setdefault(name, []).append(
                            "noisy: robust first-difference "
                            f"z={float(score):.3f} > {config.noisy_channel_zscore:g}"
                        )
            else:
                quality_warnings.append(
                    "Noisy-channel scoring was inconclusive because robust scales had no spread."
                )
        else:
            quality_warnings.append(
                "Noisy-channel scoring was skipped because too few channels had "
                "measurable variation."
            )
    else:
        quality_warnings.append(
            "Noisy-channel scoring requires at least four non-flat, non-bad EEG channels."
        )
    detected = tuple(name for name in eeg_names if name in reasons)
    raw.info["bads"] = list(dict.fromkeys([*raw.info["bads"], *detected]))
    return BadChannelResult(
        existing=existing,
        detected=detected,
        reasons={name: tuple(values) for name, values in reasons.items()},
        warnings=tuple(quality_warnings),
    )


def apply_frequency_filters(raw: BaseRaw, config: PreprocessingConfig) -> tuple[str, ...]:
    """Apply only filtering that is not already satisfied by acquisition metadata."""
    warnings_out: list[str] = []
    channel_names = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind in {"eeg", "eog"}
    ]
    active_notches = [
        value for value in config.notch_hz if config.highpass_hz < value < config.lowpass_hz
    ]
    skipped_notches = [value for value in config.notch_hz if value not in active_notches]
    if skipped_notches:
        warnings_out.append(
            "Notch frequencies outside the configured passband were skipped: "
            + ", ".join(f"{value:g} Hz" for value in skipped_notches)
            + "."
        )
    if active_notches:
        try:
            raw.notch_filter(active_notches, picks=channel_names, verbose="ERROR")
        except (RuntimeError, ValueError) as exc:
            raise PreprocessingError(f"Notch filtering failed: {exc}") from exc

    current_highpass = float(raw.info["highpass"])
    current_lowpass = float(raw.info["lowpass"])
    l_freq = config.highpass_hz if config.highpass_hz > current_highpass + 1e-9 else None
    h_freq = config.lowpass_hz if config.lowpass_hz < current_lowpass - 1e-9 else None
    if l_freq is None and h_freq is None:
        warnings_out.append(
            "Band-pass filtering was skipped because the recording already satisfies "
            "the configured passband."
        )
    else:
        try:
            raw.filter(l_freq=l_freq, h_freq=h_freq, picks=channel_names, verbose="ERROR")
        except (RuntimeError, ValueError) as exc:
            raise PreprocessingError(f"Band-pass filtering failed: {exc}") from exc
    return tuple(warnings_out)


def apply_eeg_reference(raw: BaseRaw, reference: str | tuple[str, ...]) -> tuple[str, ...]:
    """Apply average, named-channel, or explicitly disabled EEG reference."""
    if reference == "none":
        return ("EEG rereferencing was disabled by configuration.",)
    if reference == "average":
        try:
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
        except (RuntimeError, ValueError) as exc:
            raise PreprocessingError(f"Average rereferencing failed: {exc}") from exc
        return ()
    references = [reference] if isinstance(reference, str) else list(reference)
    missing = [name for name in references if name not in raw.ch_names]
    if missing:
        raise PreprocessingError("Reference channel(s) not found: " + ", ".join(missing))
    bad = [name for name in references if name in raw.info["bads"]]
    if bad:
        raise PreprocessingError(
            "Configured reference channel(s) are marked bad: " + ", ".join(bad)
        )
    try:
        raw.set_eeg_reference(ref_channels=references, projection=False, verbose="ERROR")
    except (RuntimeError, ValueError) as exc:
        raise PreprocessingError(f"EEG rereferencing failed: {exc}") from exc
    return ()


def resample_recording(raw: BaseRaw, target_hz: float | None) -> tuple[str, ...]:
    """Downsample in place using MNE's anti-alias filtering."""
    if target_hz is None:
        return ()
    current = float(raw.info["sfreq"])
    if abs(target_hz - current) <= 1e-9:
        return ("Resampling was skipped because the recording already has the target rate.",)
    try:
        raw.resample(target_hz, verbose="ERROR")
    except (RuntimeError, ValueError) as exc:
        raise PreprocessingError(f"Resampling failed: {exc}") from exc
    return ()


def preprocess_raw(raw: BaseRaw, config: PipelineConfig) -> PreprocessingResult:
    """Run configured preprocessing on a copy of a loaded MNE Raw recording."""
    original_sfreq = float(raw.info["sfreq"])
    original_eeg_count = sum(kind == "eeg" for kind in raw.get_channel_types())
    working = select_eeg_and_eog(raw, config.preprocessing.eeg_channels)
    validate_recording_settings(working, config.preprocessing)
    try:
        working.load_data()
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreprocessingError(f"Could not load EEG samples for preprocessing: {exc}") from exc

    bad_result = detect_bad_channels(working, config.artifacts)
    retained_count = sum(
        kind == "eeg" and name not in working.info["bads"]
        for name, kind in zip(working.ch_names, working.get_channel_types(), strict=True)
    )
    if retained_count == 0:
        raise PreprocessingError("No usable EEG channels remain after bad-channel handling.")

    processing_warnings: list[str] = list(bad_result.warnings)
    processing_warnings.extend(apply_frequency_filters(working, config.preprocessing))
    processing_warnings.extend(apply_eeg_reference(working, config.preprocessing.reference))
    ica_result: ICAResult = apply_ica_artifact_correction(working, config.artifacts)
    processing_warnings.extend(ica_result.warnings)
    processing_warnings.extend(resample_recording(working, config.preprocessing.resample_hz))
    try:
        epoch_result: EpochingResult = create_fixed_length_epochs(
            working, config.epochs, config.artifacts
        )
    except EpochingError as exc:
        raise PreprocessingError(str(exc)) from exc
    processing_warnings.extend(epoch_result.warnings)

    reference_value: str | list[str]
    if isinstance(config.preprocessing.reference, tuple):
        reference_value = list(config.preprocessing.reference)
    else:
        reference_value = config.preprocessing.reference
    parameters: dict[str, Any] = {
        "highpass_hz": config.preprocessing.highpass_hz,
        "lowpass_hz": config.preprocessing.lowpass_hz,
        "notch_hz": list(config.preprocessing.notch_hz),
        "resample_hz": config.preprocessing.resample_hz,
        "reference": reference_value,
        "eeg_channels": (
            None
            if config.preprocessing.eeg_channels is None
            else list(config.preprocessing.eeg_channels)
        ),
        "automatic_bad_channel_detection": config.artifacts.reject_bad_channels,
        "flat_threshold_uv": config.artifacts.flat_threshold_uv,
        "noisy_channel_zscore": config.artifacts.noisy_channel_zscore,
        "amplitude_threshold_uv": config.artifacts.amplitude_threshold_uv,
        "epoch_duration_seconds": config.epochs.duration_seconds,
        "epoch_overlap_seconds": config.epochs.overlap_seconds,
        "reject_by_annotation": config.epochs.reject_by_annotation,
        "ica": {
            "enabled": config.artifacts.ica_enabled,
            "n_components": config.artifacts.ica_n_components,
            "random_state": config.artifacts.ica_random_state,
            "max_iter": config.artifacts.ica_max_iter,
            "fit_highpass_hz": config.artifacts.ica_fit_highpass_hz,
            "eog_detection": config.artifacts.ica_eog_detection,
            "eog_threshold": config.artifacts.ica_eog_threshold,
            "minimum_eeg_channels": config.artifacts.ica_min_eeg_channels,
            "minimum_duration_seconds": config.artifacts.ica_min_duration_seconds,
        },
    }
    return PreprocessingResult(
        raw=working,
        epochs=epoch_result.epochs,
        original_sampling_frequency_hz=original_sfreq,
        final_sampling_frequency_hz=float(working.info["sfreq"]),
        original_eeg_channel_count=original_eeg_count,
        retained_eeg_channel_count=retained_count,
        existing_bad_channels=bad_result.existing,
        detected_bad_channels=bad_result.detected,
        bad_channel_reasons=bad_result.reasons,
        ica_status=ica_result.status,
        ica_candidate_components=ica_result.candidate_components,
        ica_removed_components=ica_result.removed_components,
        ica_evidence=ica_result.evidence,
        initial_epoch_count=epoch_result.initial_count,
        retained_epoch_count=epoch_result.retained_count,
        epoch_rejection_reasons=epoch_result.rejection_reasons,
        parameters=parameters,
        warnings=tuple(processing_warnings),
    )


def save_preprocessing_result(
    result: PreprocessingResult,
    recording: Recording,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Save local FIF outputs; this does not claim BIDS-derivative compliance."""
    destination = Path(output_dir).expanduser().resolve()
    if result.retained_epoch_count == 0:
        raise PreprocessingError("No epochs were retained; preprocessing outputs were not saved.")
    destination.mkdir(parents=True, exist_ok=True)
    entities = [f"sub-{recording.participant}"]
    if recording.session:
        entities.append(f"ses-{recording.session}")
    if recording.task:
        entities.append(f"task-{recording.task}")
    if recording.acquisition:
        entities.append(f"acq-{recording.acquisition}")
    if recording.run:
        entities.append(f"run-{recording.run}")
    if recording.recording:
        entities.append(f"recording-{recording.recording}")
    stem = "_".join(entities)
    raw_path = destination / f"{stem}_desc-clean_raw.fif"
    epochs_path = destination / f"{stem}_desc-clean_epo.fif"
    if not overwrite:
        existing = [path for path in (raw_path, epochs_path) if path.exists()]
        if existing:
            raise PreprocessingError(
                "Output already exists and overwrite is disabled: "
                + ", ".join(str(path) for path in existing)
            )
    try:
        result.raw.save(raw_path, overwrite=overwrite, verbose="ERROR")
        result.epochs.save(epochs_path, overwrite=overwrite, verbose="ERROR")
    except (OSError, RuntimeError, ValueError) as exc:
        raise PreprocessingError(f"Could not save preprocessing outputs: {exc}") from exc
    return raw_path, epochs_path
