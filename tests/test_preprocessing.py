from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import config_data
from mne import Annotations, create_info
from mne.io import RawArray

from eeg_feature_pipeline.config import ArtifactConfig, PreprocessingConfig, load_config
from eeg_feature_pipeline.preprocessing import (
    PreprocessingError,
    apply_frequency_filters,
    detect_bad_channels,
    preprocess_raw,
    save_preprocessing_result,
)


def _raw(*, duration: float = 12.0, sfreq: float = 200.0, include_eog: bool = True) -> RawArray:
    rng = np.random.default_rng(7)
    times = np.arange(int(duration * sfreq)) / sfreq
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4"]
    data = np.vstack(
        [
            (5 + index) * 1e-6 * np.sin(2 * np.pi * (8 + index) * times)
            + rng.normal(0, (0.08 + 0.01 * index) * 1e-6, len(times))
            for index in range(len(names))
        ]
    )
    types = ["eeg"] * len(names)
    if include_eog:
        names.append("EOG1")
        types.append("eog")
        data = np.vstack([data, np.zeros(len(times))])
    return RawArray(data, create_info(names, sfreq, types), verbose="ERROR")


def _config(tmp_path: Path, write_config, *, mutate=None):
    dataset = tmp_path / "bids"
    dataset.mkdir(exist_ok=True)
    data = config_data(dataset)
    data["preprocessing"]["notch_hz"] = []
    data["preprocessing"]["resample_hz"] = 100.0
    data["artifacts"]["reject_bad_channels"] = False
    data["artifacts"]["reject_bad_epochs"] = False
    data["epochs"]["duration_seconds"] = 2.0
    data["epochs"]["overlap_seconds"] = 1.0
    if mutate is not None:
        mutate(data)
    return load_config(write_config(data))


def test_successful_preprocessing_and_input_immutability(tmp_path: Path, write_config) -> None:
    raw = _raw()
    raw.info["bads"] = ["C4"]
    raw.set_annotations(Annotations([0.5], [0.1], ["marker"]))
    original_data = raw.get_data().copy()
    original_bads = list(raw.info["bads"])
    original_annotations = raw.annotations.copy()

    result = preprocess_raw(raw, _config(tmp_path, write_config))

    assert result.final_sampling_frequency_hz == 100.0
    assert result.original_eeg_channel_count == 6
    assert result.retained_eeg_channel_count == 5
    assert result.existing_bad_channels == ("C4",)
    assert result.ica_status == "disabled"
    assert result.initial_epoch_count == 11
    assert result.retained_epoch_count == 11
    assert result.raw.annotations.description.tolist() == ["marker"]
    np.testing.assert_array_equal(raw.get_data(), original_data)
    assert raw.info["bads"] == original_bads
    assert raw.annotations == original_annotations


def test_filter_validation_uses_actual_sampling_frequency(tmp_path: Path, write_config) -> None:
    config = _config(
        tmp_path,
        write_config,
        mutate=lambda data: data["preprocessing"].update(resample_hz=None),
    )
    with pytest.raises(PreprocessingError, match="Nyquist"):
        preprocess_raw(_raw(sfreq=80.0), config)


def test_already_satisfied_passband_is_not_filtered_again() -> None:
    raw = _raw(duration=6.0)
    raw.load_data().filter(1.0, 45.0, verbose="ERROR")
    before = raw.get_data().copy()

    messages = apply_frequency_filters(
        raw,
        PreprocessingConfig(highpass_hz=1.0, lowpass_hz=45.0, notch_hz=()),
    )

    np.testing.assert_array_equal(raw.get_data(), before)
    assert any("already satisfies" in message for message in messages)


def test_no_eeg_channels(tmp_path: Path, write_config) -> None:
    raw = RawArray(np.zeros((1, 1000)), create_info(["EOG1"], 200.0, "eog"), verbose="ERROR")
    with pytest.raises(PreprocessingError, match="no EEG channels"):
        preprocess_raw(raw, _config(tmp_path, write_config))


def test_configured_eeg_channel_subset_is_used(tmp_path: Path, write_config) -> None:
    config = _config(
        tmp_path,
        write_config,
        mutate=lambda data: data["preprocessing"].update(eeg_channels=["Fp1", "C3"]),
    )

    result = preprocess_raw(_raw(), config)

    assert result.original_eeg_channel_count == 6
    assert result.retained_eeg_channel_count == 2
    assert result.epochs.ch_names == ["Fp1", "C3"]
    assert result.parameters["eeg_channels"] == ["Fp1", "C3"]


def test_configured_eeg_channel_must_exist(tmp_path: Path, write_config) -> None:
    config = _config(
        tmp_path,
        write_config,
        mutate=lambda data: data["preprocessing"].update(eeg_channels=["missing"]),
    )

    with pytest.raises(PreprocessingError, match="were not found"):
        preprocess_raw(_raw(), config)


def test_existing_flat_and_noisy_channels_are_distinguished() -> None:
    raw = _raw(duration=10.0)
    raw.info["bads"] = ["C4"]
    raw._data[0] = 0.0
    rng = np.random.default_rng(19)
    raw._data[4] = rng.normal(0, 100e-6, raw.n_times)

    result = detect_bad_channels(
        raw,
        config=ArtifactConfig(noisy_channel_zscore=3.0),
    )

    assert result.existing == ("C4",)
    assert "Fp1" in result.detected
    assert result.reasons["Fp1"][0].startswith("flat:")
    assert "C3" in result.detected
    assert result.reasons["C3"][0].startswith("noisy:")
    assert "C4" not in result.detected


def test_save_outputs_uses_entity_based_names(tmp_path: Path, write_config) -> None:
    result = preprocess_raw(_raw(), _config(tmp_path, write_config))
    recording = SimpleNamespace(
        participant="01",
        session=None,
        task="rest",
        acquisition=None,
        run="02",
        recording=None,
    )

    raw_path, epochs_path = save_preprocessing_result(result, recording, tmp_path / "outputs")

    assert raw_path.name == "sub-01_task-rest_run-02_desc-clean_raw.fif"
    assert epochs_path.name == "sub-01_task-rest_run-02_desc-clean_epo.fif"
    assert raw_path.is_file()
    assert epochs_path.is_file()
