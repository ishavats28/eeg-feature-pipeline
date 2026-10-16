from __future__ import annotations

from pathlib import Path

import pytest
from conftest import config_data

from eeg_feature_pipeline.config import ConfigError, load_config


def test_load_valid_config_resolves_paths(tmp_path: Path, write_config) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["dataset"]["participant"] = "sub-01"

    config = load_config(write_config(data))

    assert config.dataset.root == dataset.resolve()
    assert config.dataset.participant == "01"
    assert config.features.frequency_bands["theta"] == (4.0, 8.0)
    assert config.epochs.overlap_seconds == 2.0


def test_boolean_bids_entity_fails_with_yaml_guidance(tmp_path: Path, write_config) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["dataset"]["session"] = False

    with pytest.raises(ConfigError, match="BIDS entity value"):
        load_config(write_config(data))


def test_missing_required_input_path_fails(tmp_path: Path, write_config) -> None:
    path = write_config(config_data(tmp_path / "missing"))

    with pytest.raises(ConfigError, match="BIDS dataset root does not exist"):
        load_config(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("preprocessing", "highpass_hz", 50.0), "must be below"),
        (("preprocessing", "resample_hz", 80.0), "must exceed twice"),
        (("epochs", "overlap_seconds", 4.0), "must be less"),
        (("reliability", "segment_count", 1), "at least 2"),
    ],
)
def test_invalid_numeric_settings_fail(
    tmp_path: Path, write_config, change: tuple[str, str, object], message: str
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    section, key, value = change
    data[section][key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(data))


def test_overlapping_frequency_bands_fail(tmp_path: Path, write_config) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["features"]["frequency_bands"]["theta"] = [3.5, 8.0]

    with pytest.raises(ConfigError, match="overlap"):
        load_config(write_config(data))


def test_frequency_band_above_lowpass_fails(tmp_path: Path, write_config) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["features"]["frequency_bands"]["gamma"] = [45.0, 50.0]

    with pytest.raises(ConfigError, match="cannot extend above"):
        load_config(write_config(data))


def test_invalid_yaml_fails(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("dataset: [unterminated", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unknown_setting_fails(tmp_path: Path, write_config) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["epochs"]["duraton_seconds"] = 5.0

    with pytest.raises(ConfigError, match="Unknown setting"):
        load_config(write_config(data))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("flat_threshold_uv", 0.0, "positive"),
        ("flat_threshold_uv", 200.0, "must be below"),
        ("noisy_channel_zscore", -1.0, "positive"),
        ("ica_n_components", 1.5, "between 0 and 1"),
        ("ica_max_iter", 0, "at least 1"),
        ("ica_min_eeg_channels", 1, "at least 2"),
        ("ica_eog_detection", "yes", "true or false"),
    ],
)
def test_invalid_artifact_settings_fail(
    tmp_path: Path, write_config, key: str, value: object, message: str
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["artifacts"][key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(data))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("features", "selected", ["unknown_spectrum"], "Unsupported spectral"),
        ("features", "temporal_selected", ["median"], "Unsupported temporal"),
        ("features", "psd_window_seconds", 5.0, "cannot exceed epoch"),
        ("features", "psd_overlap_fraction", 1.0, r"\[0, 1\)"),
        ("features", "psd_n_fft", 100, "cannot be shorter"),
        ("connectivity", "methods", ["correlation"], "Unsupported connectivity"),
        ("connectivity", "bands", ["alpha"], "not defined"),
        ("connectivity", "minimum_epochs", 1, "at least 2"),
    ],
)
def test_invalid_feature_and_connectivity_settings_fail(
    tmp_path: Path,
    write_config,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data[section][key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(data))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("reliability", "method", "icc", "none or spearman"),
        ("reliability", "minimum_epochs_per_group", 0, "at least 1"),
        ("sensitivity", "filter_variants_hz", [[0.5, 45], [1, 45]], "one alternative"),
        ("reporting", "figure_dpi", 50, "at least 72"),
        ("reporting", "figure_dpi", 1000, "must not exceed"),
        ("reporting", "qc_enabled", "yes", "true or false"),
    ],
)
def test_invalid_block04_settings_fail(
    tmp_path: Path,
    write_config,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data[section][key] = value

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(data))
