from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as mpimg
import mne
import numpy as np
import pandas as pd
import pytest
from conftest import config_data

from eeg_feature_pipeline.features import FeatureExtractionResult, extract_features
from eeg_feature_pipeline.plots import PlotError, generate_diagnostic_figures
from eeg_feature_pipeline.preprocessing import preprocess_raw
from eeg_feature_pipeline.qc import collect_qc_metrics, save_qc_metrics
from eeg_feature_pipeline.reliability import (
    analyze_within_recording_stability,
    save_stability_results,
)
from eeg_feature_pipeline.sensitivity import (
    SensitivityError,
    analyze_preprocessing_sensitivity,
)


def _recording() -> SimpleNamespace:
    return SimpleNamespace(
        participant="01",
        session="baseline",
        task="rest",
        run="01",
        acquisition=None,
        recording=None,
    )


def _raw(duration: float = 14.0) -> mne.io.RawArray:
    sfreq = 200.0
    time = np.arange(int(duration * sfreq)) / sfreq
    rng = np.random.default_rng(42)
    data = np.vstack(
        [
            (5 + index) * 1e-6 * np.sin(2 * np.pi * (6 + index) * time)
            + rng.normal(0, 0.05e-6, len(time))
            for index in range(4)
        ]
    )
    return mne.io.RawArray(
        data, mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg"), verbose="ERROR"
    )


def _config(tmp_path: Path, write_config, *, sensitivity: bool = False):
    root = tmp_path / "bids"
    root.mkdir(exist_ok=True)
    data = config_data(root)
    data["preprocessing"].update(notch_hz=[], resample_hz=100.0)
    data["epochs"].update(duration_seconds=2.0, overlap_seconds=1.0)
    data["artifacts"].update(reject_bad_channels=False, reject_bad_epochs=False)
    data["reliability"]["minimum_epochs_per_group"] = 2
    data["sensitivity"].update(enabled=sensitivity, filter_variants_hz=[[0.5, 45.0]])
    from eeg_feature_pipeline.config import load_config

    return load_config(write_config(data))


def _tables(epoch_values: list[float]) -> FeatureExtractionResult:
    spectral_rows = []
    temporal_rows = []
    for epoch, value in enumerate(epoch_values):
        for channel_index, channel in enumerate(("Fz", "Cz", "Pz")):
            current = value + channel_index
            spectral_rows.append(
                {
                    "participant": "01",
                    "session": None,
                    "task": "rest",
                    "run": "01",
                    "epoch": epoch,
                    "channel": channel,
                    "frequency_band": "alpha",
                    "band_low_hz": 8.0,
                    "band_high_hz": 13.0,
                    "absolute_power_uv2": current,
                    "relative_power": current / 10,
                    "spectral_entropy": np.nan,
                }
            )
            temporal_rows.append(
                {
                    "participant": "01",
                    "session": None,
                    "task": "rest",
                    "run": "01",
                    "epoch": epoch,
                    "channel": channel,
                    "rms_uv": current,
                    "variance_uv2": current**2,
                    "hjorth_activity_uv2": current**2,
                    "hjorth_mobility": 1.0,
                    "hjorth_complexity": 1.0,
                }
            )
    return FeatureExtractionResult(
        pd.DataFrame(spectral_rows),
        pd.DataFrame(temporal_rows),
        pd.DataFrame(),
        (1.0, 2.0),
        {},
    )


def test_stability_detects_stable_and_changed_halves(tmp_path: Path, write_config) -> None:
    config = _config(tmp_path, write_config)
    stable = analyze_within_recording_stability(
        _tables([2, 2, 2, 2, 2, 2, 2, 2]), config.reliability, config.epochs
    )
    changed = analyze_within_recording_stability(
        _tables([2, 2, 2, 2, 8, 8, 8, 8]), config.reliability, config.epochs
    )

    assert stable.sufficient
    assert stable.table["absolute_difference"].max() == pytest.approx(0)
    assert changed.table["absolute_difference"].max() > 0
    assert stable.summary["groups_temporally_nonoverlapping"] is True
    assert stable.summary["boundary_epochs_omitted"]


def test_stability_insufficient_constant_and_nan_handling(tmp_path: Path, write_config) -> None:
    config = _config(tmp_path, write_config)
    insufficient = analyze_within_recording_stability(
        _tables([1, 1, 1]), config.reliability, config.epochs
    )
    constant = analyze_within_recording_stability(
        _tables([1, 1, 1, 1, 1, 1, 1, 1]), config.reliability, config.epochs
    )

    assert insufficient.summary["status"] == "insufficient_data"
    assert insufficient.table.empty
    assert any(
        item["reason"] == "constant_feature_vector"
        for item in constant.summary["rank_correlations"]
        if item["paired_observations"] >= 3
    )


def test_stability_serialization_does_not_write_empty_tsv(tmp_path: Path, write_config) -> None:
    config = _config(tmp_path, write_config)
    result = analyze_within_recording_stability(
        _tables([1, 1, 1]), config.reliability, config.epochs
    )
    paths = save_stability_results(result, _recording(), tmp_path / "out")

    assert set(paths) == {"summary"}
    assert json.loads(paths["summary"].read_text())["status"] == "insufficient_data"


def test_sensitivity_identical_and_changed_settings_preserve_source(
    tmp_path: Path, write_config
) -> None:
    config = _config(tmp_path, write_config, sensitivity=True)
    raw = _raw()
    original = raw.get_data().copy()
    identical_config = replace(
        config,
        sensitivity=replace(
            config.sensitivity,
            filter_variants_hz=(
                (config.preprocessing.highpass_hz, config.preprocessing.lowpass_hz),
            ),
        ),
    )
    identical = analyze_preprocessing_sensitivity(raw, identical_config, _recording())
    changed = analyze_preprocessing_sensitivity(raw, config, _recording())

    assert identical.table["absolute_change"].max() == pytest.approx(0, abs=1e-12)
    assert identical.comparison["parameter_differences"] == {}
    assert "filter_passband_hz" in changed.comparison["parameter_differences"]
    assert changed.comparison["comparison_scope"] == "matched"
    np.testing.assert_array_equal(raw.get_data(), original)


def test_sensitivity_rejects_inapplicable_passband(tmp_path: Path, write_config) -> None:
    config = _config(tmp_path, write_config, sensitivity=True)
    invalid = replace(
        config,
        sensitivity=replace(config.sensitivity, filter_variants_hz=((0.5, 55.0),)),
    )
    with pytest.raises(SensitivityError, match="feature analysis range|resampled Nyquist"):
        analyze_preprocessing_sensitivity(_raw(), invalid, _recording())


def test_qc_metrics_counts_flags_and_json(tmp_path: Path, write_config) -> None:
    config = _config(tmp_path, write_config)
    result = preprocess_raw(_raw(), config)
    qc = collect_qc_metrics(result)
    path = save_qc_metrics(qc, _recording(), tmp_path / "qc")
    loaded = json.loads(path.read_text())

    assert loaded["epochs"]["rejection_fraction"] == pytest.approx(0)
    assert loaded["ica"]["status"] == "disabled"
    assert loaded["bad_channels"]["flat_flags"] == []
    assert loaded["processed_signal_amplitude"]["median_channel_rms_uv"] is not None


def test_diagnostic_figures_are_nonempty_readable_and_protected(
    tmp_path: Path, write_config
) -> None:
    config = _config(tmp_path, write_config)
    raw = _raw()
    preprocessing = preprocess_raw(raw, config)
    features = extract_features(preprocessing.epochs, config, _recording())
    stability = analyze_within_recording_stability(features, config.reliability, config.epochs)
    output = tmp_path / "figures"
    figures = generate_diagnostic_figures(
        raw,
        preprocessing,
        _recording(),
        output,
        features=features,
        stability=stability,
        dpi=72,
    )

    assert {"signal", "psd", "epoch_retention", "band_power", "stability"} <= set(figures.paths)
    for path in figures.paths.values():
        assert path.stat().st_size > 0
        assert mpimg.imread(path).size > 0
    with pytest.raises(PlotError, match="overwrite is disabled"):
        generate_diagnostic_figures(raw, preprocessing, _recording(), output, features=features)
