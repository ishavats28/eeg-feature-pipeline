"""Compact diagnostic visualizations for preprocessing and quality analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import welch

from .features import FeatureExtractionResult
from .preprocessing import PreprocessingResult
from .reliability import StabilityResult
from .sensitivity import SensitivityResult

if TYPE_CHECKING:
    from mne.io import BaseRaw


class PlotError(RuntimeError):
    """Raised when diagnostic figures cannot be saved safely."""


@dataclass(frozen=True)
class PlotResult:
    """Created figure paths and nonfatal omissions."""

    paths: dict[str, Path]
    warnings: tuple[str, ...]


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


def _common_eeg_channel(original: BaseRaw, processed: BaseRaw) -> str | None:
    original_eeg = {
        name
        for name, kind in zip(original.ch_names, original.get_channel_types(), strict=True)
        if kind == "eeg" and name not in original.info["bads"]
    }
    for name, kind in zip(processed.ch_names, processed.get_channel_types(), strict=True):
        if kind == "eeg" and name not in processed.info["bads"] and name in original_eeg:
            return name
    return None


def _save(fig: Figure, path: Path, dpi: int) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, format="png", bbox_inches="tight")
    finally:
        fig.clear()


def generate_diagnostic_figures(
    original: BaseRaw,
    preprocessing: PreprocessingResult,
    recording: Any,
    output_dir: str | Path,
    *,
    features: FeatureExtractionResult | None = None,
    stability: StabilityResult | None = None,
    sensitivity: SensitivityResult | None = None,
    dpi: int = 150,
    overwrite: bool = False,
) -> PlotResult:
    """Generate a modest set of actual-data PNG diagnostics.

    Before and processed traces use independent time axes. PSD curves use the
    intersection of available frequency coverage and no smoothing beyond Welch's
    estimator.
    """
    destination = Path(output_dir).expanduser().resolve()
    stem = _entity_stem(recording)
    paths: dict[str, Path] = {}
    warnings: list[str] = []
    channel = _common_eeg_channel(original, preprocessing.raw)
    if channel is not None:
        paths["signal"] = destination / f"{stem}_desc-qc_signal.png"
        paths["psd"] = destination / f"{stem}_desc-qc_psd.png"
    else:
        warnings.append("No common eligible EEG channel was available for signal and PSD plots.")
    paths["epoch_retention"] = destination / f"{stem}_desc-qc_epoch-retention.png"
    if features is not None and not features.spectral_table.empty:
        band_rows = features.spectral_table[
            features.spectral_table["frequency_band"] != "analysis_range"
        ]
        if not band_rows.empty and band_rows["absolute_power_uv2"].notna().any():
            paths["band_power"] = destination / f"{stem}_desc-qc_band-power.png"
        else:
            warnings.append("Band-power figure omitted because absolute band power is unavailable.")
    if (
        stability is not None
        and not stability.table.empty
        and stability.table["relative_difference"].notna().any()
    ):
        paths["stability"] = destination / f"{stem}_desc-stability_comparison.png"
    elif stability is not None:
        warnings.append("Stability figure omitted because no matched stability values exist.")
    if (
        sensitivity is not None
        and not sensitivity.table.empty
        and sensitivity.table["relative_change"].notna().any()
    ):
        paths["sensitivity"] = destination / f"{stem}_desc-sensitivity_comparison.png"
    elif sensitivity is not None:
        warnings.append("Sensitivity figure omitted because no matched sensitivity values exist.")

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise PlotError(
            "Diagnostic figure already exists and overwrite is disabled: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    try:
        if channel is not None:
            original_signal = original.get_data(picks=[channel])[0] * 1e6
            processed_signal = preprocessing.raw.get_data(picks=[channel])[0] * 1e6
            fig = Figure(figsize=(9, 5))
            axes = fig.subplots(2, 1, sharex=False)
            for axis, signal, raw, label, color in (
                (axes[0], original_signal, original, "Before preprocessing", "#4c78a8"),
                (
                    axes[1],
                    processed_signal,
                    preprocessing.raw,
                    "After preprocessing",
                    "#f58518",
                ),
            ):
                count = min(len(signal), int(round(10 * float(raw.info["sfreq"]))))
                times = np.arange(count) / float(raw.info["sfreq"])
                axis.plot(times, signal[:count], color=color, linewidth=0.7)
                axis.set_ylabel("Amplitude (µV)")
                axis.set_title(f"{label}: {channel}")
            axes[-1].set_xlabel("Time within displayed segment (s)")
            fig.suptitle("Independent trace views; samples are not asserted to be aligned")
            _save(fig, paths["signal"], dpi)

            spectra: list[tuple[np.ndarray, np.ndarray, str, str]] = []
            for signal, raw, label, color in (
                (original_signal, original, "Before preprocessing", "#4c78a8"),
                (processed_signal, preprocessing.raw, "After preprocessing", "#f58518"),
            ):
                sfreq = float(raw.info["sfreq"])
                nperseg = min(len(signal), max(8, int(round(2 * sfreq))))
                frequency, density = welch(
                    signal,
                    fs=sfreq,
                    window="hamming",
                    nperseg=nperseg,
                    noverlap=nperseg // 2,
                    scaling="density",
                )
                spectra.append((frequency, density, label, color))
            maximum = min(
                float(preprocessing.parameters.get("lowpass_hz") or np.inf),
                spectra[0][0][-1],
                spectra[1][0][-1],
            )
            fig = Figure(figsize=(8, 4.5))
            axis = fig.subplots()
            for frequency, density, label, color in spectra:
                mask = (frequency > 0) & (frequency <= maximum) & (density > 0)
                axis.semilogy(frequency[mask], density[mask], label=label, color=color)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("PSD (µV²/Hz)")
            axis.set_title(f"Welch PSD comparison: {channel}")
            axis.legend()
            _save(fig, paths["psd"], dpi)

        fig = Figure(figsize=(6.5, 4))
        axis = fig.subplots()
        initial = preprocessing.initial_epoch_count
        retained = preprocessing.retained_epoch_count
        axis.bar(
            ["Initial", "Rejected", "Retained"],
            [initial, initial - retained, retained],
            color=["#4c78a8", "#e45756", "#54a24b"],
        )
        axis.set_ylabel("Epoch count")
        axis.set_title("Epoch retention")
        _save(fig, paths["epoch_retention"], dpi)

        if "band_power" in paths and features is not None:
            band_rows = features.spectral_table[
                features.spectral_table["frequency_band"] != "analysis_range"
            ]
            summary = band_rows.groupby("frequency_band", sort=False)["absolute_power_uv2"].median()
            fig = Figure(figsize=(7, 4))
            axis = fig.subplots()
            axis.bar(summary.index.astype(str), summary.values, color="#4c78a8")
            axis.set_ylabel("Median absolute power (µV²)")
            axis.set_xlabel("Frequency band")
            axis.set_title("EEG band-power summary across epochs and channels")
            _save(fig, paths["band_power"], dpi)

        if "stability" in paths and stability is not None:
            data = stability.table.dropna(subset=["relative_difference"])
            summary = data.groupby("feature", sort=True)["relative_difference"].median() * 100
            fig = Figure(figsize=(8, 4.5))
            axis = fig.subplots()
            axis.bar(summary.index, summary.values, color="#4c78a8")
            axis.tick_params(axis="x", rotation=35)
            axis.set_ylabel("Median absolute relative difference (%)")
            axis.set_title("Within-recording feature stability")
            _save(fig, paths["stability"], dpi)

        if "sensitivity" in paths and sensitivity is not None:
            data = sensitivity.table.dropna(subset=["relative_change"])
            summary = data.groupby("feature", sort=True)["relative_change"].median() * 100
            fig = Figure(figsize=(8, 4.5))
            axis = fig.subplots()
            colors = np.where(summary.values >= 0, "#f58518", "#4c78a8")
            axis.bar(summary.index, summary.values, color=colors)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.tick_params(axis="x", rotation=35)
            axis.set_ylabel("Median relative change (%)")
            axis.set_title("Preprocessing sensitivity (matched feature rows)")
            _save(fig, paths["sensitivity"], dpi)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PlotError(f"Could not generate diagnostic figures: {exc}") from exc
    return PlotResult(paths, tuple(warnings))
