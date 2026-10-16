"""Spectral and temporal features from cleaned resting-state EEG epochs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch
from scipy.integrate import trapezoid

from .config import FeatureConfig, PipelineConfig
from .connectivity import ConnectivityResult, compute_connectivity

if TYPE_CHECKING:
    from mne import Epochs


class FeatureExtractionError(RuntimeError):
    """Raised when configured features cannot be computed safely."""


@dataclass(frozen=True)
class PSDResult:
    """Welch density estimates in V²/Hz with their frequency bins."""

    values_v2_per_hz: np.ndarray
    frequencies_hz: np.ndarray
    channel_order: tuple[str, ...]
    n_fft: int
    n_per_segment: int
    n_overlap: int


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Tidy feature tables plus reproducibility diagnostics."""

    spectral_table: pd.DataFrame
    temporal_table: pd.DataFrame
    connectivity_table: pd.DataFrame
    psd_frequencies_hz: tuple[float, ...]
    diagnostics: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "spectral_rows": len(self.spectral_table),
            "temporal_rows": len(self.temporal_table),
            "connectivity_rows": len(self.connectivity_table),
            "psd_frequency_bin_count": len(self.psd_frequencies_hz),
            "diagnostics": self.diagnostics,
        }


def _metadata(recording: Any) -> dict[str, str | None]:
    return {
        "participant": getattr(recording, "participant", None),
        "session": getattr(recording, "session", None),
        "task": getattr(recording, "task", None),
        "run": getattr(recording, "run", None),
    }


def _epoch_identifiers(epochs: Epochs) -> tuple[int, ...]:
    selection = getattr(epochs, "selection", np.arange(len(epochs)))
    return tuple(int(value) for value in selection)


def compute_welch_psd(epochs: Epochs, config: FeatureConfig) -> PSDResult:
    """Compute per-epoch, per-channel Welch PSD using a Hamming window."""
    if len(epochs) == 0:
        raise FeatureExtractionError("No retained epochs are available for PSD estimation.")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude="bads")
    if len(picks) == 0:
        raise FeatureExtractionError("No usable EEG channels are available for PSD estimation.")
    data = epochs.get_data(picks=picks, units="V", copy=True)
    if not np.all(np.isfinite(data)):
        raise FeatureExtractionError("PSD input contains nonfinite EEG samples.")
    sfreq = float(epochs.info["sfreq"])
    nyquist = sfreq / 2.0
    low, high = config.analysis_range_hz
    if not 0 <= low < high < nyquist:
        raise FeatureExtractionError(
            f"PSD analysis range {low:g}-{high:g} Hz must lie below Nyquist ({nyquist:g} Hz)."
        )
    n_per_segment = int(round(config.psd_window_seconds * sfreq))
    if n_per_segment < 2:
        raise FeatureExtractionError("Welch window corresponds to fewer than two samples.")
    if n_per_segment > data.shape[-1]:
        raise FeatureExtractionError(
            f"Welch window requires {n_per_segment} samples but epochs contain "
            f"{data.shape[-1]}; settings will not be silently truncated."
        )
    n_fft = n_per_segment if config.psd_n_fft is None else config.psd_n_fft
    if n_fft < n_per_segment:
        raise FeatureExtractionError(
            f"Welch n_fft ({n_fft}) must be at least the window length ({n_per_segment})."
        )
    n_overlap = int(round(config.psd_overlap_fraction * n_per_segment))
    if n_overlap >= n_per_segment:
        raise FeatureExtractionError("Welch overlap must be shorter than the Welch window.")
    try:
        psd, frequencies = psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=low,
            fmax=high,
            n_fft=n_fft,
            n_overlap=n_overlap,
            n_per_seg=n_per_segment,
            average="mean",
            window="hamming",
            remove_dc=True,
            n_jobs=1,
            verbose="ERROR",
        )
    except (RuntimeError, ValueError) as exc:
        raise FeatureExtractionError(f"Welch PSD estimation failed: {exc}") from exc
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if psd.shape != (len(epochs), len(picks), len(frequencies)):
        raise FeatureExtractionError(f"Unexpected PSD output shape: {psd.shape}.")
    if not np.all(np.isfinite(psd)) or np.any(psd < 0):
        raise FeatureExtractionError("Welch PSD produced negative or nonfinite density values.")
    if len(frequencies) < 2:
        raise FeatureExtractionError("PSD analysis range contains fewer than two frequency bins.")
    return PSDResult(
        values_v2_per_hz=psd,
        frequencies_hz=frequencies,
        channel_order=tuple(epochs.ch_names[index] for index in picks),
        n_fft=n_fft,
        n_per_segment=n_per_segment,
        n_overlap=n_overlap,
    )


def extract_spectral_features(
    epochs: Epochs,
    config: FeatureConfig,
    recording: Any,
) -> tuple[pd.DataFrame, PSDResult, dict[str, Any]]:
    """Extract band power and normalized base-2 spectral entropy.

    Absolute power is trapezoidal integration of Welch density and is reported in
    µV². Relative power uses integrated power across the full configured analysis
    range as its denominator, not the sum of selected band powers.
    """
    psd = compute_welch_psd(epochs, config)
    frequencies = psd.frequencies_hz
    analysis_power = trapezoid(psd.values_v2_per_hz, frequencies, axis=-1)
    near_zero = analysis_power <= 1e-30
    probability_denominator = np.sum(psd.values_v2_per_hz, axis=-1)
    entropy_invalid = probability_denominator <= 1e-30
    probabilities = np.divide(
        psd.values_v2_per_hz,
        probability_denominator[..., np.newaxis],
        out=np.zeros_like(psd.values_v2_per_hz),
        where=~entropy_invalid[..., np.newaxis],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probabilities > 0, probabilities * np.log2(probabilities), 0.0)
    entropy = -np.sum(entropy_terms, axis=-1) / np.log2(len(frequencies))
    entropy[entropy_invalid] = np.nan

    selected = set(config.selected)
    compute_absolute = "band_power" in selected or "absolute_band_power" in selected
    compute_relative = "band_power" in selected or "relative_band_power" in selected
    compute_entropy = "spectral_entropy" in selected
    metadata = _metadata(recording)
    epoch_ids = _epoch_identifiers(epochs)
    rows: list[dict[str, Any]] = []
    for epoch_position, epoch_id in enumerate(epoch_ids):
        for channel_index, channel_name in enumerate(psd.channel_order):
            if compute_absolute or compute_relative:
                for band_name, (low, high) in config.frequency_bands.items():
                    mask = (frequencies >= low) & (frequencies <= high)
                    if np.count_nonzero(mask) < 2:
                        raise FeatureExtractionError(
                            f"Frequency band '{band_name}' contains fewer than two Welch bins."
                        )
                    band_power_v2 = float(
                        trapezoid(
                            psd.values_v2_per_hz[epoch_position, channel_index, mask],
                            frequencies[mask],
                        )
                    )
                    relative = np.nan
                    if not near_zero[epoch_position, channel_index]:
                        relative = band_power_v2 / analysis_power[epoch_position, channel_index]
                    rows.append(
                        {
                            **metadata,
                            "epoch": epoch_id,
                            "channel": channel_name,
                            "frequency_band": band_name,
                            "band_low_hz": low,
                            "band_high_hz": high,
                            "absolute_power_uv2": (
                                band_power_v2 * 1e12 if compute_absolute else np.nan
                            ),
                            "relative_power": relative if compute_relative else np.nan,
                            "spectral_entropy": np.nan,
                        }
                    )
            if compute_entropy:
                rows.append(
                    {
                        **metadata,
                        "epoch": epoch_id,
                        "channel": channel_name,
                        "frequency_band": "analysis_range",
                        "band_low_hz": config.analysis_range_hz[0],
                        "band_high_hz": config.analysis_range_hz[1],
                        "absolute_power_uv2": np.nan,
                        "relative_power": np.nan,
                        "spectral_entropy": float(entropy[epoch_position, channel_index]),
                    }
                )
    columns = [
        "participant",
        "session",
        "task",
        "run",
        "epoch",
        "channel",
        "frequency_band",
        "band_low_hz",
        "band_high_hz",
        "absolute_power_uv2",
        "relative_power",
        "spectral_entropy",
    ]
    diagnostics = {
        "welch": {
            "window": "hamming",
            "window_seconds": config.psd_window_seconds,
            "overlap_fraction": config.psd_overlap_fraction,
            "n_per_segment": psd.n_per_segment,
            "n_overlap": psd.n_overlap,
            "n_fft": psd.n_fft,
            "density_units": "V^2/Hz",
            "integration": "scipy.integrate.trapezoid",
        },
        "analysis_range_hz": list(config.analysis_range_hz),
        "undefined_relative_power_count": int(np.count_nonzero(near_zero)),
        "undefined_spectral_entropy_count": int(np.count_nonzero(entropy_invalid)),
    }
    return pd.DataFrame(rows, columns=columns), psd, diagnostics


def _temporal_metrics(signal_uv: np.ndarray) -> tuple[dict[str, float], tuple[str, ...]]:
    names = ("rms", "variance", "hjorth_activity", "hjorth_mobility", "hjorth_complexity")
    undefined = {name: np.nan for name in names}
    if not np.all(np.isfinite(signal_uv)):
        return undefined, ("nonfinite_input",)
    if len(signal_uv) == 0:
        return undefined, ("empty_signal",)
    rms = float(np.sqrt(np.mean(np.square(signal_uv))))
    activity = float(np.var(signal_uv, ddof=0))
    metrics = {
        "rms": rms,
        "variance": activity,
        "hjorth_activity": activity,
        "hjorth_mobility": np.nan,
        "hjorth_complexity": np.nan,
    }
    reasons: list[str] = []
    if len(signal_uv) < 2 or activity <= np.finfo(float).eps:
        reasons.append("mobility_undefined")
        reasons.append("complexity_undefined")
        return metrics, tuple(reasons)
    first_difference = np.diff(signal_uv)
    first_variance = float(np.var(first_difference, ddof=0))
    if first_variance <= np.finfo(float).eps:
        reasons.extend(("mobility_undefined", "complexity_undefined"))
        return metrics, tuple(reasons)
    mobility = float(np.sqrt(first_variance / activity))
    metrics["hjorth_mobility"] = mobility
    if len(signal_uv) < 3 or mobility <= np.finfo(float).eps:
        reasons.append("complexity_undefined")
        return metrics, tuple(reasons)
    second_variance = float(np.var(np.diff(first_difference), ddof=0))
    derivative_mobility = np.sqrt(second_variance / first_variance)
    complexity = float(derivative_mobility / mobility)
    if np.isfinite(complexity):
        metrics["hjorth_complexity"] = complexity
    else:
        reasons.append("complexity_undefined")
    return metrics, tuple(reasons)


def extract_temporal_features(
    epochs: Epochs,
    config: FeatureConfig,
    recording: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract microvolt-domain RMS, population variance, and Hjorth parameters."""
    if len(epochs) == 0:
        raise FeatureExtractionError("No retained epochs are available for temporal features.")
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude="bads")
    if len(picks) == 0:
        raise FeatureExtractionError("No usable EEG channels are available for temporal features.")
    data_uv = epochs.get_data(picks=picks, units="uV", copy=True)
    channels = tuple(epochs.ch_names[index] for index in picks)
    epoch_ids = _epoch_identifiers(epochs)
    metadata = _metadata(recording)
    selected = set(config.temporal_selected)
    rows: list[dict[str, Any]] = []
    undefined_counts: dict[str, int] = {}
    for epoch_position, epoch_id in enumerate(epoch_ids):
        for channel_index, channel_name in enumerate(channels):
            metrics, reasons = _temporal_metrics(data_uv[epoch_position, channel_index])
            for reason in reasons:
                undefined_counts[reason] = undefined_counts.get(reason, 0) + 1
            rows.append(
                {
                    **metadata,
                    "epoch": epoch_id,
                    "channel": channel_name,
                    **{
                        name: value if name in selected else np.nan
                        for name, value in metrics.items()
                    },
                }
            )
    columns = [
        "participant",
        "session",
        "task",
        "run",
        "epoch",
        "channel",
        "rms",
        "variance",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
    ]
    table = pd.DataFrame(rows, columns=columns).rename(
        columns={
            "rms": "rms_uv",
            "variance": "variance_uv2",
            "hjorth_activity": "hjorth_activity_uv2",
        }
    )
    return table, {
        "amplitude_units": "microvolts",
        "variance_convention": "population variance (ddof=0)",
        "derivative_method": "first discrete sample difference",
        "undefined_metric_counts": dict(sorted(undefined_counts.items())),
    }


def extract_features(
    epochs: Epochs,
    config: PipelineConfig,
    recording: Any,
) -> FeatureExtractionResult:
    """Extract configured spectral, temporal, and optional connectivity tables."""
    spectral, psd, spectral_diagnostics = extract_spectral_features(
        epochs, config.features, recording
    )
    temporal, temporal_diagnostics = extract_temporal_features(epochs, config.features, recording)
    connectivity: ConnectivityResult = compute_connectivity(
        epochs,
        config.connectivity,
        config.features.frequency_bands,
        recording,
    )
    return FeatureExtractionResult(
        spectral_table=spectral,
        temporal_table=temporal,
        connectivity_table=connectivity.table,
        psd_frequencies_hz=tuple(float(value) for value in psd.frequencies_hz),
        diagnostics={
            "spectral": spectral_diagnostics,
            "temporal": temporal_diagnostics,
            "connectivity": {
                "enabled": config.connectivity.enabled,
                "analysis_level": "recording_pooled_epochs",
                "channel_order": list(connectivity.channel_order),
                "epoch_count": connectivity.epoch_count,
                "warnings": list(connectivity.warnings),
            },
        },
    )


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


def save_feature_tables(
    result: FeatureExtractionResult,
    recording: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save non-empty tidy TSV tables to a local analysis-output directory."""
    tables = {
        "spectral": result.spectral_table,
        "temporal": result.temporal_table,
    }
    if not result.connectivity_table.empty:
        tables["connectivity"] = result.connectivity_table
    if result.spectral_table.empty or result.temporal_table.empty:
        raise FeatureExtractionError("Required feature tables are empty; no files were written.")
    destination = Path(output_dir).expanduser().resolve()
    paths = {
        name: destination / f"{_entity_stem(recording)}_desc-{name}_features.tsv" for name in tables
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FeatureExtractionError(
                "Feature output already exists and overwrite is disabled: "
                + ", ".join(str(path) for path in existing)
            )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for name, table in tables.items():
            table.to_csv(paths[name], sep="\t", index=False, na_rep="n/a")
    except OSError as exc:
        raise FeatureExtractionError(f"Could not save feature tables: {exc}") from exc
    return paths
