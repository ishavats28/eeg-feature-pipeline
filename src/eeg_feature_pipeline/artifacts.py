"""Conservative ICA artifact management for EEG preprocessing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from mne.preprocessing import ICA

from .config import ArtifactConfig

if TYPE_CHECKING:
    from mne.io import BaseRaw


@dataclass(frozen=True)
class ICAResult:
    """JSON-friendly diagnostics from optional ICA fitting and application."""

    status: str
    candidate_components: tuple[int, ...] = ()
    removed_components: tuple[int, ...] = ()
    evidence: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()


def _warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [f"{item.category.__name__}: {item.message}" for item in caught]


def apply_ica_artifact_correction(raw: BaseRaw, config: ArtifactConfig) -> ICAResult:
    """Fit guarded EEG ICA and apply only EOG-supported component exclusions.

    The supplied ``raw`` object is modified only when EOG-informed candidates are
    identified. ICA is fitted to a separate copy that is high-pass filtered to at
    least ``ica_fit_highpass_hz``; correction is applied to the processing signal.
    """
    if not config.ica_enabled:
        return ICAResult(status="disabled")

    eeg_names = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind == "eeg" and name not in raw.info["bads"]
    ]
    if len(eeg_names) < config.ica_min_eeg_channels:
        return ICAResult(
            status="skipped_insufficient_channels",
            warnings=(
                f"ICA requires at least {config.ica_min_eeg_channels} usable EEG channels; "
                f"found {len(eeg_names)}.",
            ),
        )
    duration = raw.n_times / float(raw.info["sfreq"])
    if duration < config.ica_min_duration_seconds:
        return ICAResult(
            status="skipped_insufficient_samples",
            warnings=(
                f"ICA requires at least {config.ica_min_duration_seconds:g} seconds; "
                f"found {duration:.3f} seconds.",
            ),
        )
    if isinstance(config.ica_n_components, int) and config.ica_n_components >= len(eeg_names):
        return ICAResult(
            status="skipped_invalid_component_count",
            warnings=(
                "Integer ICA component count must be smaller than the number of usable "
                f"EEG channels ({len(eeg_names)}) after bad-channel exclusion.",
            ),
        )

    fit_raw = raw.copy()
    fit_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if float(fit_raw.info["highpass"]) < config.ica_fit_highpass_hz:
                fit_raw.filter(
                    l_freq=config.ica_fit_highpass_hz,
                    h_freq=None,
                    picks=eeg_names,
                    verbose="ERROR",
                )
            ica = ICA(
                n_components=config.ica_n_components,
                random_state=config.ica_random_state,
                max_iter=config.ica_max_iter,
                method="fastica",
                verbose="ERROR",
            )
            ica.fit(
                fit_raw,
                picks=eeg_names,
                reject_by_annotation=True,
                verbose="ERROR",
            )
        fit_warnings.extend(_warning_messages(caught))
    except Exception as exc:
        return ICAResult(
            status="failed",
            warnings=tuple([*fit_warnings, f"ICA fitting failed: {type(exc).__name__}: {exc}"]),
        )

    eog_names = [
        name
        for name, kind in zip(fit_raw.ch_names, fit_raw.get_channel_types(), strict=True)
        if kind == "eog" and name not in fit_raw.info["bads"]
    ]
    evidence: dict[str, Any] = {
        "selection_method": "mne_find_bads_eog_zscore",
        "eog_channels": eog_names,
        "component_scores": {},
        "threshold": config.ica_eog_threshold,
    }
    if not config.ica_eog_detection:
        return ICAResult(
            status="fitted_selection_disabled",
            evidence=evidence,
            warnings=tuple(fit_warnings),
        )
    if not eog_names:
        return ICAResult(
            status="fitted_no_eog",
            evidence=evidence,
            warnings=tuple(
                [*fit_warnings, "No valid EOG channel was available for ICA selection."]
            ),
        )

    candidates: set[int] = set()
    scores_by_component: dict[str, float] = {}
    selection_warnings: list[str] = []
    successful_selections = 0
    for eog_name in eog_names:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                indices, scores = ica.find_bads_eog(
                    fit_raw,
                    ch_name=eog_name,
                    threshold=config.ica_eog_threshold,
                    reject_by_annotation=True,
                    verbose="ERROR",
                )
            selection_warnings.extend(_warning_messages(caught))
        except Exception as exc:
            selection_warnings.append(
                f"EOG component selection failed for {eog_name}: {type(exc).__name__}: {exc}"
            )
            continue
        successful_selections += 1
        score_values = np.asarray(scores, dtype=float).ravel()
        for index in indices:
            component = int(index)
            candidates.add(component)
            if component < len(score_values):
                key = str(component)
                scores_by_component[key] = max(
                    scores_by_component.get(key, 0.0), abs(float(score_values[component]))
                )
    evidence["component_scores"] = scores_by_component
    ordered_candidates = tuple(sorted(candidates))
    all_warnings = tuple([*fit_warnings, *selection_warnings])
    if successful_selections == 0:
        return ICAResult(
            status="fitted_selection_failed",
            evidence=evidence,
            warnings=all_warnings,
        )
    if not ordered_candidates:
        return ICAResult(
            status="fitted_no_components",
            evidence=evidence,
            warnings=all_warnings,
        )

    try:
        ica.apply(raw, exclude=list(ordered_candidates), verbose="ERROR")
    except Exception as exc:
        return ICAResult(
            status="failed_application",
            candidate_components=ordered_candidates,
            evidence=evidence,
            warnings=tuple([*all_warnings, f"ICA application failed: {type(exc).__name__}: {exc}"]),
        )
    return ICAResult(
        status="applied",
        candidate_components=ordered_candidates,
        removed_components=ordered_candidates,
        evidence=evidence,
        warnings=all_warnings,
    )
