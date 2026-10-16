"""Lightweight one-parameter preprocessing sensitivity analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .features import FeatureExtractionResult, extract_spectral_features, extract_temporal_features
from .preprocessing import PreprocessingResult, preprocess_raw

if TYPE_CHECKING:
    from mne.io import BaseRaw


class SensitivityError(RuntimeError):
    """Raised when the configured sensitivity comparison is invalid or fails."""


@dataclass(frozen=True)
class SensitivityResult:
    """Matched feature changes plus the two preprocessing branch results."""

    table: pd.DataFrame
    comparison: dict[str, Any]
    baseline: PreprocessingResult
    alternative: PreprocessingResult


_METADATA = ["participant", "session", "task", "run"]
_SPECTRAL_VALUES = ["absolute_power_uv2", "relative_power", "spectral_entropy"]
_TEMPORAL_VALUES = [
    "rms_uv",
    "variance_uv2",
    "hjorth_activity_uv2",
    "hjorth_mobility",
    "hjorth_complexity",
]


def _feature_rows(spectral: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in _SPECTRAL_VALUES:
        frame = spectral[[*_METADATA, "epoch", "channel", "frequency_band", name]].copy()
        frame["feature_family"] = "spectral"
        frame["feature"] = name
        frame["value"] = pd.to_numeric(frame.pop(name), errors="coerce")
        frames.append(frame)
    for name in _TEMPORAL_VALUES:
        frame = temporal[[*_METADATA, "epoch", "channel", name]].copy()
        frame["frequency_band"] = pd.NA
        frame["feature_family"] = "temporal"
        frame["feature"] = name
        frame["value"] = pd.to_numeric(frame.pop(name), errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _eligible_channels(result: PreprocessingResult) -> list[str]:
    bads = set(result.raw.info["bads"])
    return sorted(
        name
        for name, kind in zip(result.raw.ch_names, result.raw.get_channel_types(), strict=True)
        if kind == "eeg" and name not in bads
    )


def _branch_features(
    result: PreprocessingResult, config: PipelineConfig, recording: Any
) -> pd.DataFrame:
    if result.retained_epoch_count == 0:
        raise SensitivityError("A sensitivity branch retained zero epochs.")
    spectral, _, _ = extract_spectral_features(result.epochs, config.features, recording)
    temporal, _ = extract_temporal_features(result.epochs, config.features, recording)
    return _feature_rows(spectral, temporal)


def analyze_preprocessing_sensitivity(
    raw: BaseRaw,
    config: PipelineConfig,
    recording: Any,
    *,
    baseline_result: PreprocessingResult | None = None,
    baseline_features: FeatureExtractionResult | None = None,
) -> SensitivityResult:
    """Compare baseline preprocessing with one alternative filter passband.

    Both branches start independently from the supplied Raw object. The source is
    not modified. Only exactly matched epoch/channel/feature rows are compared;
    differing retained epoch or channel sets are labelled descriptive/confounded.
    Connectivity is intentionally excluded.
    """
    if not config.sensitivity.enabled:
        raise SensitivityError("Preprocessing sensitivity analysis is disabled.")
    if len(config.sensitivity.filter_variants_hz) != 1:
        raise SensitivityError("Sensitivity analysis requires one alternative passband.")
    alternative_low, alternative_high = config.sensitivity.filter_variants_hz[0]
    nyquist = float(raw.info["sfreq"]) / 2.0
    if alternative_high >= nyquist:
        raise SensitivityError(
            f"Alternative low-pass must be below the source Nyquist frequency ({nyquist:g} Hz)."
        )
    if alternative_low < float(raw.info["highpass"]) - 1e-9:
        raise SensitivityError(
            "Alternative high-pass is below the source recording high-pass and cannot be restored."
        )
    if alternative_high < config.features.analysis_range_hz[1]:
        raise SensitivityError(
            "Alternative low-pass cannot be below the configured feature analysis range."
        )
    if config.preprocessing.resample_hz is not None and (
        config.preprocessing.resample_hz / 2.0 <= alternative_high
    ):
        raise SensitivityError(
            "Alternative low-pass must be below the resampled Nyquist frequency."
        )

    alternative_preprocessing = replace(
        config.preprocessing,
        highpass_hz=alternative_low,
        lowpass_hz=alternative_high,
    )
    alternative_config = replace(config, preprocessing=alternative_preprocessing)
    baseline = baseline_result if baseline_result is not None else preprocess_raw(raw, config)
    alternative = preprocess_raw(raw, alternative_config)
    baseline_rows = (
        _feature_rows(
            baseline_features.spectral_table,
            baseline_features.temporal_table,
        )
        if baseline_features is not None
        else _branch_features(baseline, config, recording)
    )
    alternative_rows = _branch_features(alternative, alternative_config, recording)

    keys = [
        *_METADATA,
        "epoch",
        "channel",
        "frequency_band",
        "feature_family",
        "feature",
    ]
    table = baseline_rows.merge(
        alternative_rows,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_alternative"),
        validate="one_to_one",
    )
    finite = np.isfinite(table["value_baseline"]) & np.isfinite(table["value_alternative"])
    table = table.loc[finite].copy()
    table["absolute_change"] = (table["value_alternative"] - table["value_baseline"]).abs()
    baseline_magnitude = table["value_baseline"].abs()
    table["relative_change"] = (table["value_alternative"] - table["value_baseline"]).div(
        baseline_magnitude.where(baseline_magnitude > np.finfo(float).eps)
    )

    baseline_epochs = sorted(int(value) for value in baseline_rows["epoch"].unique())
    alternative_epochs = sorted(int(value) for value in alternative_rows["epoch"].unique())
    baseline_channels = _eligible_channels(baseline)
    alternative_channels = _eligible_channels(alternative)
    fully_matched = (
        baseline_epochs == alternative_epochs and baseline_channels == alternative_channels
    )
    pairing = "matched" if fully_matched else "partially_matched_descriptive"
    table["comparison_scope"] = pairing
    table = table.sort_values(
        ["feature_family", "feature", "epoch", "channel", "frequency_band"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    baseline_passband = [
        config.preprocessing.highpass_hz,
        config.preprocessing.lowpass_hz,
    ]
    alternative_passband = [alternative_low, alternative_high]
    differences: dict[str, Any] = {}
    if baseline_passband != alternative_passband:
        differences["filter_passband_hz"] = {
            "baseline": baseline_passband,
            "alternative": alternative_passband,
        }
    warnings: list[str] = []
    if not differences:
        warnings.append("Baseline and alternative preprocessing settings are identical.")
    if not fully_matched:
        warnings.append(
            "Retained epoch or channel sets differ; matched rows are descriptive "
            "and potentially confounded."
        )
    summaries: list[dict[str, Any]] = []
    for (family, feature), frame in table.groupby(["feature_family", "feature"], sort=True):
        relative = frame["relative_change"].dropna()
        summaries.append(
            {
                "feature_family": family,
                "feature": feature,
                "matched_rows": len(frame),
                "median_absolute_change": float(frame["absolute_change"].median()),
                "median_relative_change": (
                    float(relative.median()) if not relative.empty else None
                ),
            }
        )
    comparison = {
        "analysis": "preprocessing sensitivity",
        "interpretation": "descriptive; not a measure of biological accuracy",
        "status": "ok" if not table.empty else "insufficient_data",
        "comparison_scope": pairing,
        "parameter_differences": differences,
        "baseline_preprocessing": asdict(config.preprocessing),
        "alternative_preprocessing": asdict(alternative_preprocessing),
        "baseline": {
            "retained_epochs": baseline.retained_epoch_count,
            "eligible_channels": baseline_channels,
            "epoch_ids": baseline_epochs,
        },
        "alternative": {
            "retained_epochs": alternative.retained_epoch_count,
            "eligible_channels": alternative_channels,
            "epoch_ids": alternative_epochs,
        },
        "matched_feature_rows": len(table),
        "descriptive_summaries": summaries,
        "undefined_relative_change_count": int(table["relative_change"].isna().sum()),
        "warnings": warnings,
    }
    if table.empty:
        comparison["warnings"].append("No finite feature rows matched between branches.")
    return SensitivityResult(table, comparison, baseline, alternative)


def analyze_sensitivity(
    raw: BaseRaw,
    config: PipelineConfig,
    recording: Any,
) -> SensitivityResult:
    """Alias for the one-parameter preprocessing sensitivity analysis."""
    return analyze_preprocessing_sensitivity(raw, config, recording)


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


def save_sensitivity_results(
    result: SensitivityResult,
    recording: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save a nonempty comparison TSV and configuration comparison JSON."""
    if result.table.empty:
        raise SensitivityError("Sensitivity comparison is empty; no result files were written.")
    destination = Path(output_dir).expanduser().resolve()
    stem = _entity_stem(recording)
    paths = {
        "features": destination / f"{stem}_desc-sensitivity_features.tsv",
        "configuration": destination / f"{stem}_desc-sensitivity_config.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SensitivityError(
            "Sensitivity output already exists and overwrite is disabled: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        result.table.to_csv(paths["features"], sep="\t", index=False, na_rep="n/a")
        paths["configuration"].write_text(
            json.dumps(result.comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        raise SensitivityError(f"Could not save sensitivity results: {exc}") from exc
    return paths
