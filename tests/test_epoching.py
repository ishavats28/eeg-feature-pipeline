from __future__ import annotations

import numpy as np
import pytest
from mne import Annotations, create_info
from mne.io import RawArray

from eeg_feature_pipeline.config import ArtifactConfig, EpochConfig
from eeg_feature_pipeline.epoching import EpochingError, create_fixed_length_epochs


def _raw(duration: float = 10.0, sfreq: float = 100.0) -> RawArray:
    times = np.arange(int(duration * sfreq)) / sfreq
    data = np.vstack(
        [
            10e-6 * np.sin(2 * np.pi * 10 * times),
            8e-6 * np.sin(2 * np.pi * 12 * times),
        ]
    )
    return RawArray(data, create_info(["Fz", "Cz"], sfreq, "eeg"), verbose="ERROR")


def test_epoch_duration_and_overlap() -> None:
    result = create_fixed_length_epochs(
        _raw(),
        EpochConfig(duration_seconds=2.0, overlap_seconds=1.0),
        ArtifactConfig(reject_bad_channels=False, reject_bad_epochs=False),
    )

    assert result.initial_count == 9
    assert result.retained_count == 9
    assert result.epochs.baseline is None
    assert result.epochs.get_data().shape[-1] == 200


def test_bad_annotation_excludes_overlapping_epochs() -> None:
    raw = _raw()
    raw.set_annotations(Annotations([2.25], [1.0], ["BAD_movement"]))

    result = create_fixed_length_epochs(
        raw,
        EpochConfig(duration_seconds=2.0, overlap_seconds=0.0, reject_by_annotation=True),
        ArtifactConfig(reject_bad_channels=False, reject_bad_epochs=False),
    )

    assert result.initial_count == 5
    assert result.retained_count == 4
    assert result.rejection_reasons["BAD_movement"] == 1


def test_amplitude_rejection_records_channel_reason() -> None:
    raw = _raw()
    raw._data[0, 450:550] = 500e-6

    result = create_fixed_length_epochs(
        raw,
        EpochConfig(duration_seconds=2.0, overlap_seconds=0.0),
        ArtifactConfig(
            reject_bad_channels=False,
            reject_bad_epochs=True,
            amplitude_threshold_uv=100.0,
        ),
    )

    assert result.retained_count == 4
    assert result.rejection_reasons["Fz"] == 1


def test_zero_retained_epochs_is_explicit() -> None:
    raw = _raw()
    raw._data[0] *= 100

    result = create_fixed_length_epochs(
        raw,
        EpochConfig(duration_seconds=2.0, overlap_seconds=0.0),
        ArtifactConfig(
            reject_bad_channels=False,
            reject_bad_epochs=True,
            amplitude_threshold_uv=1.0,
        ),
    )

    assert result.initial_count == 5
    assert result.retained_count == 0
    assert result.warnings == ("All fixed-length epochs were rejected.",)


def test_recording_too_short_is_explicit() -> None:
    with pytest.raises(EpochingError, match="too short"):
        create_fixed_length_epochs(
            _raw(duration=1.0),
            EpochConfig(duration_seconds=2.0, overlap_seconds=0.0),
            ArtifactConfig(reject_bad_channels=False, reject_bad_epochs=False),
        )
