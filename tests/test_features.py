from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest
from conftest import config_data
from scipy.integrate import trapezoid

from eeg_feature_pipeline.config import FeatureConfig, load_config
from eeg_feature_pipeline.features import (
    FeatureExtractionError,
    compute_welch_psd,
    extract_features,
    extract_spectral_features,
    extract_temporal_features,
    save_feature_tables,
)
from eeg_feature_pipeline.preprocessing import preprocess_raw


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(
        participant="01",
        session="baseline",
        task="rest",
        run="01",
        acquisition=None,
        recording=None,
    )


def _alpha_epochs(*, constant: bool = False) -> mne.EpochsArray:
    sfreq = 100.0
    times = np.arange(400) / sfreq
    if constant:
        epoch = np.zeros((2, len(times)))
    else:
        epoch = np.vstack(
            [
                10e-6 * np.sin(2 * np.pi * 10 * times),
                5e-6 * np.sin(2 * np.pi * 6 * times),
            ]
        )
    data = np.stack([epoch for _ in range(3)])
    return mne.EpochsArray(
        data,
        mne.create_info(["Alpha", "Theta"], sfreq, "eeg"),
        verbose="ERROR",
    )


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        frequency_bands={"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
        selected=("absolute_band_power", "relative_band_power", "spectral_entropy"),
        analysis_range_hz=(4.0, 30.0),
        psd_window_seconds=2.0,
        psd_overlap_fraction=0.5,
    )


def test_welch_frequency_resolution_and_power_normalization() -> None:
    epochs = _alpha_epochs()
    psd = compute_welch_psd(epochs, _feature_config())

    assert psd.values_v2_per_hz.shape == (3, 2, 53)
    assert np.diff(psd.frequencies_hz).mean() == pytest.approx(0.5)
    alpha_index = psd.channel_order.index("Alpha")
    integrated_uv2 = np.trapezoid(psd.values_v2_per_hz[0, alpha_index], psd.frequencies_hz) * 1e12
    assert integrated_uv2 == pytest.approx(50.0, rel=0.15)


def test_alpha_dominance_relative_power_and_entropy_bounds() -> None:
    table, psd, diagnostics = extract_spectral_features(
        _alpha_epochs(), _feature_config(), _metadata()
    )
    alpha_channel = table[table["channel"] == "Alpha"]
    means = alpha_channel.groupby("frequency_band")["absolute_power_uv2"].mean()

    assert means["alpha"] > 20 * means["theta"]
    assert means["alpha"] > 20 * means["beta"]
    alpha_rows = alpha_channel[alpha_channel["frequency_band"] == "alpha"]
    entropy_rows = alpha_channel[alpha_channel["frequency_band"] == "analysis_range"]
    assert np.all(alpha_rows["relative_power"] > 0.9)
    frequencies = psd.frequencies_hz
    alpha_mask = (frequencies >= 8.0) & (frequencies <= 13.0)
    expected_relative = trapezoid(
        psd.values_v2_per_hz[0, 0, alpha_mask], frequencies[alpha_mask]
    ) / trapezoid(psd.values_v2_per_hz[0, 0], frequencies)
    assert alpha_rows.iloc[0]["relative_power"] == pytest.approx(expected_relative)
    assert np.all((entropy_rows["spectral_entropy"] >= 0) & (entropy_rows["spectral_entropy"] <= 1))
    assert diagnostics["welch"]["density_units"] == "V^2/Hz"


def test_constant_signal_has_explicit_undefined_spectral_metrics() -> None:
    table, _, diagnostics = extract_spectral_features(
        _alpha_epochs(constant=True), _feature_config(), _metadata()
    )

    assert table["relative_power"].isna().all()
    assert table["spectral_entropy"].isna().all()
    assert diagnostics["undefined_relative_power_count"] == 6
    assert diagnostics["undefined_spectral_entropy_count"] == 6


def test_invalid_psd_frequency_range_and_fft_are_rejected() -> None:
    invalid_range = FeatureConfig(
        frequency_bands={"alpha": (8.0, 13.0)},
        analysis_range_hz=(1.0, 50.0),
        psd_window_seconds=2.0,
    )
    with pytest.raises(FeatureExtractionError, match="Nyquist"):
        compute_welch_psd(_alpha_epochs(), invalid_range)

    short_fft = FeatureConfig(
        frequency_bands={"alpha": (8.0, 13.0)},
        analysis_range_hz=(1.0, 20.0),
        psd_window_seconds=2.0,
        psd_n_fft=100,
    )
    with pytest.raises(FeatureExtractionError, match="at least the window"):
        compute_welch_psd(_alpha_epochs(), short_fft)


def test_temporal_known_units_and_hjorth_values() -> None:
    values_uv = np.tile(np.array([1.0, -1.0]), 50)
    epochs = mne.EpochsArray(
        values_uv[np.newaxis, np.newaxis, :] * 1e-6,
        mne.create_info(["Cz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    table, diagnostics = extract_temporal_features(epochs, _feature_config(), _metadata())
    row = table.iloc[0]
    first_variance = np.var(np.diff(values_uv), ddof=0)
    expected_mobility = np.sqrt(first_variance / np.var(values_uv, ddof=0))
    expected_complexity = (
        np.sqrt(np.var(np.diff(values_uv, n=2), ddof=0) / first_variance) / expected_mobility
    )

    assert row["rms_uv"] == pytest.approx(1.0)
    assert row["variance_uv2"] == pytest.approx(1.0)
    assert row["hjorth_activity_uv2"] == pytest.approx(1.0)
    assert row["hjorth_mobility"] == pytest.approx(expected_mobility)
    assert row["hjorth_complexity"] == pytest.approx(expected_complexity)
    assert diagnostics["variance_convention"] == "population variance (ddof=0)"


def test_temporal_constant_short_and_nonfinite_signals() -> None:
    info = mne.create_info(["Cz"], 100.0, "eeg")
    constant = mne.EpochsArray(np.ones((1, 1, 4)) * 2e-6, info, verbose="ERROR")
    constant_table, constant_diagnostics = extract_temporal_features(
        constant, _feature_config(), _metadata()
    )
    assert constant_table.iloc[0]["rms_uv"] == pytest.approx(2.0)
    assert constant_table.iloc[0]["variance_uv2"] == pytest.approx(0.0)
    assert np.isnan(constant_table.iloc[0]["hjorth_mobility"])
    assert constant_diagnostics["undefined_metric_counts"]["mobility_undefined"] == 1

    short = mne.EpochsArray(np.array([[[1e-6, -1e-6]]]), info, verbose="ERROR")
    short_table, short_diagnostics = extract_temporal_features(
        short, _feature_config(), _metadata()
    )
    assert np.isnan(short_table.iloc[0]["hjorth_mobility"])
    assert np.isnan(short_table.iloc[0]["hjorth_complexity"])
    assert short_diagnostics["undefined_metric_counts"]["mobility_undefined"] == 1
    assert short_diagnostics["undefined_metric_counts"]["complexity_undefined"] == 1

    nonfinite = mne.EpochsArray(np.array([[[1e-6, np.nan, 2e-6]]]), info, verbose="ERROR")
    nonfinite_table, nonfinite_diagnostics = extract_temporal_features(
        nonfinite, _feature_config(), _metadata()
    )
    assert nonfinite_table.iloc[0].loc["rms_uv":"hjorth_complexity"].isna().all()
    assert nonfinite_diagnostics["undefined_metric_counts"]["nonfinite_input"] == 1


def test_preprocessed_epochs_metadata_and_tsv_serialization(tmp_path: Path, write_config) -> None:
    sfreq = 200.0
    times = np.arange(int(8 * sfreq)) / sfreq
    raw = mne.io.RawArray(
        np.vstack([5e-6 * np.sin(2 * np.pi * frequency * times) for frequency in (6, 10, 12, 18)]),
        mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    dataset = tmp_path / "bids"
    dataset.mkdir()
    data = config_data(dataset)
    data["preprocessing"].update(notch_hz=[], resample_hz=100.0)
    data["epochs"].update(duration_seconds=2.0, overlap_seconds=1.0)
    data["features"].update(
        frequency_bands={"theta": [4.0, 8.0], "alpha": [8.0, 13.0]},
        analysis_range_hz=[4.0, 13.0],
    )
    data["connectivity"].update(
        enabled=True,
        methods=["coherence_squared"],
        bands=["alpha"],
        minimum_epochs=2,
    )
    data["artifacts"].update(reject_bad_channels=False, reject_bad_epochs=False)
    config = load_config(write_config(data))

    preprocessed = preprocess_raw(raw, config)
    result = extract_features(preprocessed.epochs, config, _metadata())
    paths = save_feature_tables(result, _metadata(), tmp_path / "features")

    assert set(paths) == {"spectral", "temporal", "connectivity"}
    spectral = pd.read_csv(paths["spectral"], sep="\t")
    temporal = pd.read_csv(paths["temporal"], sep="\t")
    connectivity = pd.read_csv(paths["connectivity"], sep="\t")
    assert spectral["participant"].astype(str).str.zfill(2).eq("01").all()
    assert spectral["session"].eq("baseline").all()
    assert temporal["task"].eq("rest").all()
    assert connectivity["analysis_level"].eq("recording_pooled_epochs").all()
    assert list(spectral["epoch"].unique()) == list(range(7))
