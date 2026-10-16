"""Pooled sensor-space functional connectivity from cleaned EEG epochs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

from .config import ConnectivityConfig

if TYPE_CHECKING:
    from mne import Epochs


class ConnectivityError(RuntimeError):
    """Raised when configured connectivity cannot be estimated defensibly."""


@dataclass(frozen=True)
class ConnectivityResult:
    """Tidy pooled connectivity estimates and diagnostics."""

    table: pd.DataFrame
    channel_order: tuple[str, ...]
    epoch_count: int
    warnings: tuple[str, ...] = ()


_METHOD_MAP = {
    "coherence_squared": ("coh", "magnitude_squared_coherence"),
    "pli": ("pli", "pli"),
    "wpli2_debiased": ("wpli2_debiased", "wpli2_debiased"),
}


def _metadata(recording: Any) -> dict[str, str | None]:
    return {
        "participant": getattr(recording, "participant", None),
        "session": getattr(recording, "session", None),
        "task": getattr(recording, "task", None),
        "run": getattr(recording, "run", None),
    }


def compute_connectivity(
    epochs: Epochs,
    config: ConnectivityConfig,
    frequency_bands: dict[str, tuple[float, float]],
    recording: Any,
) -> ConnectivityResult:
    """Estimate one pooled connectivity value per pair, method, and band.

    MNE's ``coh`` estimator is magnitude coherence; it is squared here to produce
    magnitude-squared coherence. PLI and debiased squared wPLI are used as returned
    by MNE-Connectivity. All estimates pool cross-spectral information across the
    supplied epochs and are not per-epoch observations.
    """
    columns = [
        "participant",
        "session",
        "task",
        "run",
        "analysis_level",
        "epoch_count",
        "source_channel",
        "target_channel",
        "frequency_band",
        "band_low_hz",
        "band_high_hz",
        "frequency_aggregation",
        "method",
        "value",
    ]
    if not config.enabled:
        return ConnectivityResult(
            table=pd.DataFrame(columns=columns),
            channel_order=(),
            epoch_count=len(epochs),
            warnings=("Connectivity was disabled by configuration.",),
        )
    if len(epochs) < config.minimum_epochs:
        raise ConnectivityError(
            f"Connectivity requires at least {config.minimum_epochs} retained epochs; "
            f"found {len(epochs)}."
        )
    picks = mne.pick_types(epochs.info, eeg=True, eog=False, exclude="bads")
    if len(picks) < 2:
        raise ConnectivityError(
            f"Connectivity requires at least two usable EEG channels; found {len(picks)}."
        )
    channel_names = tuple(epochs.ch_names[index] for index in picks)
    data = epochs.get_data(picks=picks, units="V", copy=True)
    if not np.all(np.isfinite(data)):
        raise ConnectivityError("Connectivity input contains nonfinite EEG samples.")
    sfreq = float(epochs.info["sfreq"])
    nyquist = sfreq / 2.0
    bands: list[tuple[str, float, float]] = []
    diagnostic_warnings: list[str] = []
    epoch_duration = data.shape[-1] / sfreq
    for name in config.bands:
        low, high = frequency_bands[name]
        if not 0 < low < high < nyquist:
            raise ConnectivityError(
                f"Connectivity band '{name}' ({low:g}-{high:g} Hz) must lie below "
                f"Nyquist ({nyquist:g} Hz)."
            )
        cycles = low * epoch_duration
        if cycles < 2:
            raise ConnectivityError(
                f"Connectivity band '{name}' has fewer than two cycles of its lower "
                "frequency in each epoch."
            )
        if cycles < 5:
            diagnostic_warnings.append(
                f"Connectivity band '{name}' has only {cycles:.2f} cycles of its lower "
                "frequency per epoch; interpret estimates cautiously."
            )
        bands.append((name, low, high))
    pair_indices = np.triu_indices(len(channel_names), k=0 if config.include_diagonal else 1)
    if len(pair_indices[0]) == 0:
        raise ConnectivityError("No channel pairs are available for connectivity estimation.")

    metadata = _metadata(recording)
    rows: list[dict[str, Any]] = []
    fmin = tuple(low for _, low, _ in bands)
    fmax = tuple(high for _, _, high in bands)
    for configured_method in config.methods:
        mne_method, output_label = _METHOD_MAP[configured_method]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimate = spectral_connectivity_epochs(
                    data,
                    names=list(channel_names),
                    method=mne_method,
                    indices=pair_indices,
                    sfreq=sfreq,
                    mode="multitaper",
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    mt_adaptive=False,
                    n_jobs=1,
                    verbose="ERROR",
                )
            diagnostic_warnings.extend(
                f"{item.category.__name__}: {item.message}" for item in caught
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ConnectivityError(
                f"Connectivity estimation failed for '{configured_method}': {exc}"
            ) from exc
        values = np.asarray(estimate.get_data(), dtype=float)
        expected_shape = (len(pair_indices[0]), len(bands))
        if values.shape != expected_shape:
            raise ConnectivityError(
                f"Unexpected connectivity shape {values.shape}; expected {expected_shape}."
            )
        if configured_method == "coherence_squared":
            values = np.square(values)
        if not np.all(np.isfinite(values)):
            raise ConnectivityError(
                f"Connectivity method '{configured_method}' produced nonfinite values."
            )
        for pair_position, (source_index, target_index) in enumerate(
            zip(*pair_indices, strict=True)
        ):
            for band_index, (band_name, low, high) in enumerate(bands):
                rows.append(
                    {
                        **metadata,
                        "analysis_level": "recording_pooled_epochs",
                        "epoch_count": len(epochs),
                        "source_channel": channel_names[int(source_index)],
                        "target_channel": channel_names[int(target_index)],
                        "frequency_band": band_name,
                        "band_low_hz": low,
                        "band_high_hz": high,
                        "frequency_aggregation": "band_average",
                        "method": output_label,
                        "value": float(values[pair_position, band_index]),
                    }
                )
    return ConnectivityResult(
        table=pd.DataFrame(rows, columns=columns),
        channel_order=channel_names,
        epoch_count=len(epochs),
        warnings=tuple(diagnostic_warnings),
    )
