"""Structured quantitative quality-control diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np

from .preprocessing import PreprocessingResult


class QCError(RuntimeError):
    """Raised when QC diagnostics cannot be collected or saved."""


@dataclass(frozen=True)
class QCResult:
    """JSON-serializable quality-control metrics."""

    metrics: dict[str, Any]


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _signal_summary(result: PreprocessingResult) -> dict[str, Any]:
    picks = mne.pick_types(result.raw.info, eeg=True, eog=False, exclude="bads")
    if len(picks) == 0:
        return {
            "eligible_channel_count": 0,
            "median_channel_rms_uv": None,
            "minimum_channel_rms_uv": None,
            "maximum_channel_rms_uv": None,
            "nonfinite_sample_count": None,
        }
    data_uv = result.raw.get_data(picks=picks) * 1e6
    finite = np.isfinite(data_uv)
    rms = np.full(len(picks), np.nan)
    for index, channel in enumerate(data_uv):
        values = channel[np.isfinite(channel)]
        if values.size:
            rms[index] = np.sqrt(np.mean(np.square(values)))
    return {
        "eligible_channel_count": len(picks),
        "median_channel_rms_uv": _finite_or_none(float(np.nanmedian(rms))),
        "minimum_channel_rms_uv": _finite_or_none(float(np.nanmin(rms))),
        "maximum_channel_rms_uv": _finite_or_none(float(np.nanmax(rms))),
        "nonfinite_sample_count": int(finite.size - np.count_nonzero(finite)),
        "description": "Descriptive processed-signal amplitude; not a physiological quality score.",
    }


def collect_qc_metrics(result: PreprocessingResult) -> QCResult:
    """Collect explicit preprocessing and signal diagnostics from a result."""
    reasons = {
        name: list(channel_reasons)
        for name, channel_reasons in sorted(result.bad_channel_reasons.items())
    }
    flat = sorted(
        name
        for name, channel_reasons in reasons.items()
        if any(reason.startswith("flat:") for reason in channel_reasons)
    )
    noisy = sorted(
        name
        for name, channel_reasons in reasons.items()
        if any(reason.startswith("noisy:") for reason in channel_reasons)
    )
    initial = result.initial_epoch_count
    rejected = initial - result.retained_epoch_count
    rejection_fraction = rejected / initial if initial > 0 else None
    parameters = result.parameters
    metrics: dict[str, Any] = {
        "sampling_frequency_hz": {
            "original": result.original_sampling_frequency_hz,
            "final": result.final_sampling_frequency_hz,
        },
        "eeg_channels": {
            "original": result.original_eeg_channel_count,
            "retained_eligible": result.retained_eeg_channel_count,
        },
        "bad_channels": {
            "existing": list(result.existing_bad_channels),
            "newly_detected": list(result.detected_bad_channels),
            "flat_flags": flat,
            "noisy_flags": noisy,
            "reasons": reasons,
        },
        "filtering": {
            "highpass_hz": parameters.get("highpass_hz"),
            "lowpass_hz": parameters.get("lowpass_hz"),
            "notch_hz": parameters.get("notch_hz"),
            "resample_hz": parameters.get("resample_hz"),
        },
        "reference": parameters.get("reference"),
        "ica": {
            "status": result.ica_status,
            "candidate_components": list(result.ica_candidate_components),
            "excluded_components": list(result.ica_removed_components),
            "evidence": result.ica_evidence,
        },
        "epochs": {
            "original": initial,
            "rejected": rejected,
            "retained": result.retained_epoch_count,
            "rejection_fraction": rejection_fraction,
            "rejection_reasons": dict(sorted(result.epoch_rejection_reasons.items())),
        },
        "processed_signal_amplitude": _signal_summary(result),
        "warnings": list(result.warnings),
    }
    return QCResult(metrics)


def _entity_stem(recording: Any) -> str:
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


def save_qc_metrics(
    result: QCResult,
    recording: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save QC metrics as strict JSON."""
    destination = Path(output_dir).expanduser().resolve()
    path = destination / f"{_entity_stem(recording)}_desc-qc_metrics.json"
    if path.exists() and not overwrite:
        raise QCError(f"QC output already exists and overwrite is disabled: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(result.metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        raise QCError(f"Could not save QC metrics: {exc}") from exc
    return path
