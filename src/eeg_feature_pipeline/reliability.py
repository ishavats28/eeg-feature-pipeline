"""Within-recording feature stability across disjoint temporal partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import EpochConfig, ReliabilityConfig
from .features import FeatureExtractionResult


class ReliabilityError(RuntimeError):
    """Raised when within-recording stability cannot be analyzed or saved."""


@dataclass(frozen=True)
class StabilityResult:
    """Tidy matched-feature comparisons and JSON-friendly diagnostics."""

    table: pd.DataFrame
    summary: dict[str, Any]

    @property
    def sufficient(self) -> bool:
        return self.summary["status"] == "ok"


_METADATA = ["participant", "session", "task", "run"]
_SPECTRAL_VALUES = ["absolute_power_uv2", "relative_power", "spectral_entropy"]
_TEMPORAL_VALUES = [
    "rms_uv",
    "variance_uv2",
    "hjorth_activity_uv2",
    "hjorth_mobility",
    "hjorth_complexity",
]
_OUTPUT_COLUMNS = [
    *_METADATA,
    "feature_family",
    "channel",
    "frequency_band",
    "feature",
    "group_1_mean",
    "group_2_mean",
    "absolute_difference",
    "relative_difference",
    "group_1_epoch_count",
    "group_2_epoch_count",
]


def _temporal_groups(
    epoch_ids: list[int], epochs: EpochConfig
) -> tuple[list[int], list[int], list[int]]:
    """Split epochs around recording midpoint without overlap across groups."""
    ordered = sorted(set(epoch_ids))
    if not ordered:
        return [], [], []
    step = epochs.duration_seconds - epochs.overlap_seconds
    epoch_width_in_steps = epochs.duration_seconds / step
    midpoint = (ordered[0] + ordered[-1] + epoch_width_in_steps) / 2.0
    first = [value for value in ordered if value + epoch_width_in_steps <= midpoint]
    second = [value for value in ordered if value >= midpoint]
    omitted = [value for value in ordered if value not in set(first) | set(second)]
    return first, second, omitted


def _long_feature_table(features: FeatureExtractionResult) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    spectral_keys = [*_METADATA, "epoch", "channel", "frequency_band"]
    for name in _SPECTRAL_VALUES:
        if name in features.spectral_table:
            frame = features.spectral_table[spectral_keys + [name]].copy()
            frame["feature_family"] = "spectral"
            frame["feature"] = name
            frame["value"] = pd.to_numeric(frame.pop(name), errors="coerce")
            frames.append(frame)
    temporal_keys = [*_METADATA, "epoch", "channel"]
    for name in _TEMPORAL_VALUES:
        if name in features.temporal_table:
            frame = features.temporal_table[temporal_keys + [name]].copy()
            frame["frequency_band"] = pd.NA
            frame["feature_family"] = "temporal"
            frame["feature"] = name
            frame["value"] = pd.to_numeric(frame.pop(name), errors="coerce")
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_within_recording_stability(
    features: FeatureExtractionResult,
    reliability: ReliabilityConfig,
    epochs: EpochConfig,
) -> StabilityResult:
    """Compare mean features in two chronological, nonoverlapping time groups.

    Absolute difference is ``abs(group_2 - group_1)``. Relative difference is
    that absolute difference divided by ``abs(group_1)`` and is undefined when
    the first-group mean is numerically zero.
    """
    if not reliability.enabled:
        raise ReliabilityError("Within-recording feature stability is disabled.")
    if reliability.segment_count != 2:
        raise ReliabilityError("Within-recording stability currently requires two groups.")
    long = _long_feature_table(features)
    epoch_ids = sorted(int(value) for value in long.get("epoch", pd.Series(dtype=int)).unique())
    first, second, omitted = _temporal_groups(epoch_ids, epochs)
    summary: dict[str, Any] = {
        "analysis": "within-recording feature stability",
        "status": "ok",
        "group_1_epochs": first,
        "group_2_epochs": second,
        "group_1_epoch_count": len(first),
        "group_2_epoch_count": len(second),
        "boundary_epochs_omitted": omitted,
        "configured_epoch_overlap_seconds": epochs.overlap_seconds,
        "groups_temporally_nonoverlapping": True,
        "within_group_epochs_may_overlap": epochs.overlap_seconds > 0,
        "correlation_method": reliability.method,
        "rank_correlations": [],
        "descriptive_summaries": [],
        "warnings": [],
    }
    minimum = reliability.minimum_epochs_per_group
    if len(first) < minimum or len(second) < minimum:
        summary["status"] = "insufficient_data"
        summary["warnings"].append(
            f"Each temporal group requires at least {minimum} retained epochs."
        )
        return StabilityResult(pd.DataFrame(columns=_OUTPUT_COLUMNS), summary)

    keys = [*_METADATA, "feature_family", "channel", "frequency_band", "feature"]
    grouped: list[pd.DataFrame] = []
    for label, ids in (("group_1", first), ("group_2", second)):
        subset = long[long["epoch"].isin(ids)].dropna(subset=["value"])
        means = (
            subset.groupby(keys, dropna=False, sort=True)["value"].mean().rename(f"{label}_mean")
        )
        counts = (
            subset.groupby(keys, dropna=False, sort=True)["epoch"]
            .nunique()
            .rename(f"{label}_epoch_count")
        )
        grouped.append(pd.concat([means, counts], axis=1).reset_index())
    comparison = grouped[0].merge(grouped[1], on=keys, how="inner", validate="one_to_one")
    valid = np.isfinite(comparison["group_1_mean"]) & np.isfinite(comparison["group_2_mean"])
    comparison = comparison.loc[valid].copy()
    comparison["absolute_difference"] = (
        comparison["group_2_mean"] - comparison["group_1_mean"]
    ).abs()
    denominator = comparison["group_1_mean"].abs()
    comparison["relative_difference"] = comparison["absolute_difference"].div(
        denominator.where(denominator > np.finfo(float).eps)
    )
    comparison = (
        comparison[_OUTPUT_COLUMNS]
        .sort_values(
            ["feature_family", "feature", "channel", "frequency_band"],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )
    summary["matched_feature_count"] = len(comparison)
    summary["undefined_relative_difference_count"] = int(
        comparison["relative_difference"].isna().sum()
    )
    if comparison.empty:
        summary["status"] = "insufficient_data"
        summary["warnings"].append("No finite features were matched between temporal groups.")
        return StabilityResult(comparison, summary)

    for (family, feature), frame in comparison.groupby(["feature_family", "feature"], sort=True):
        relative = frame["relative_difference"].dropna()
        summary["descriptive_summaries"].append(
            {
                "feature_family": family,
                "feature": feature,
                "matched_observations": len(frame),
                "mean_absolute_difference": float(frame["absolute_difference"].mean()),
                "median_absolute_difference": float(frame["absolute_difference"].median()),
                "median_relative_difference": (
                    float(relative.median()) if not relative.empty else None
                ),
            }
        )

    if reliability.method == "spearman":
        for (family, feature), frame in comparison.groupby(
            ["feature_family", "feature"], sort=True
        ):
            x = frame["group_1_mean"].to_numpy(float)
            y = frame["group_2_mean"].to_numpy(float)
            entry: dict[str, Any] = {
                "feature_family": family,
                "feature": feature,
                "paired_observations": len(frame),
                "correlation": None,
                "reason": None,
            }
            if len(frame) < 3:
                entry["reason"] = "fewer_than_three_pairs"
            elif np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
                entry["reason"] = "constant_feature_vector"
            else:
                correlation = float(spearmanr(x, y).statistic)
                if np.isfinite(correlation):
                    entry["correlation"] = correlation
                else:
                    entry["reason"] = "nonfinite_correlation"
            summary["rank_correlations"].append(entry)
    return StabilityResult(comparison, summary)


def analyze_stability(
    features: FeatureExtractionResult,
    reliability: ReliabilityConfig,
    epochs: EpochConfig,
) -> StabilityResult:
    """Alias with a concise name for within-recording stability analysis."""
    return analyze_within_recording_stability(features, reliability, epochs)


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


def save_stability_results(
    result: StabilityResult,
    recording: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save stability JSON and, only when nonempty, the tidy TSV."""
    destination = Path(output_dir).expanduser().resolve()
    stem = _entity_stem(recording)
    paths = {"summary": destination / f"{stem}_desc-stability_summary.json"}
    if not result.table.empty:
        paths["features"] = destination / f"{stem}_desc-stability_features.tsv"
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ReliabilityError(
            "Stability output already exists and overwrite is disabled: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        paths["summary"].write_text(
            json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if "features" in paths:
            result.table.to_csv(paths["features"], sep="\t", index=False, na_rep="n/a")
    except (OSError, ValueError) as exc:
        raise ReliabilityError(f"Could not save stability results: {exc}") from exc
    return paths
