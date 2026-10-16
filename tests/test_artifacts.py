from __future__ import annotations

import numpy as np
from mne import create_info
from mne.io import RawArray

from eeg_feature_pipeline.artifacts import apply_ica_artifact_correction
from eeg_feature_pipeline.config import ArtifactConfig


def _ica_raw(*, eeg_count: int = 6, eog: bool = False, duration: float = 8.0) -> RawArray:
    sfreq = 100.0
    times = np.arange(int(duration * sfreq)) / sfreq
    rng = np.random.default_rng(42)
    sources = [
        8e-6 * np.sin(2 * np.pi * (8 + index) * times) + rng.normal(0, 0.2e-6, len(times))
        for index in range(eeg_count)
    ]
    names = [f"EEG{index + 1}" for index in range(eeg_count)]
    types = ["eeg"] * eeg_count
    if eog:
        blink = np.zeros_like(times)
        blink[200:220] = 150e-6
        sources.append(blink)
        names.append("EOG1")
        types.append("eog")
    return RawArray(np.asarray(sources), create_info(names, sfreq, types), verbose="ERROR")


def test_ica_disabled() -> None:
    result = apply_ica_artifact_correction(_ica_raw(), ArtifactConfig(ica_enabled=False))
    assert result.status == "disabled"
    assert result.removed_components == ()


def test_ica_insufficient_channels() -> None:
    config = ArtifactConfig(
        ica_enabled=True,
        ica_min_eeg_channels=4,
        ica_min_duration_seconds=1.0,
    )
    result = apply_ica_artifact_correction(_ica_raw(eeg_count=2), config)
    assert result.status == "skipped_insufficient_channels"
    assert result.removed_components == ()


def test_ica_insufficient_samples() -> None:
    config = ArtifactConfig(
        ica_enabled=True,
        ica_min_eeg_channels=4,
        ica_min_duration_seconds=20.0,
    )
    result = apply_ica_artifact_correction(_ica_raw(duration=2.0), config)
    assert result.status == "skipped_insufficient_samples"
    assert result.removed_components == ()


def test_ica_without_eog_does_not_remove_components() -> None:
    config = ArtifactConfig(
        ica_enabled=True,
        ica_min_duration_seconds=2.0,
        ica_eog_detection=True,
    )
    result = apply_ica_artifact_correction(_ica_raw(), config)
    assert result.status == "fitted_no_eog"
    assert result.removed_components == ()


def test_ica_with_eog_records_selection_evidence() -> None:
    config = ArtifactConfig(
        ica_enabled=True,
        ica_min_duration_seconds=2.0,
        ica_eog_detection=True,
        ica_eog_threshold=100.0,
    )
    result = apply_ica_artifact_correction(_ica_raw(eog=True), config)
    assert result.status == "fitted_no_components"
    assert result.evidence is not None
    assert result.evidence["eog_channels"] == ["EOG1"]
    assert result.removed_components == ()
