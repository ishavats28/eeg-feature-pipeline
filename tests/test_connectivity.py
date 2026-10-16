from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from eeg_feature_pipeline.config import ConnectivityConfig
from eeg_feature_pipeline.connectivity import ConnectivityError, compute_connectivity


def _epochs(*, epoch_count: int = 6, channel_count: int = 3) -> mne.EpochsArray:
    sfreq = 100.0
    times = np.arange(400) / sfreq
    rng = np.random.default_rng(21)
    data = []
    for _ in range(epoch_count):
        shared = 10e-6 * np.sin(2 * np.pi * 10 * times)
        channels = [shared, 10e-6 * np.sin(2 * np.pi * 10 * times + 0.4)]
        if channel_count > 2:
            channels.extend(rng.normal(0, 7e-6, len(times)) for _ in range(channel_count - 2))
        data.append(np.asarray(channels))
    return mne.EpochsArray(
        np.asarray(data),
        mne.create_info([f"EEG{index + 1}" for index in range(channel_count)], sfreq, "eeg"),
        verbose="ERROR",
    )


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(participant="01", session=None, task="rest", run="01")


def test_connectivity_coupling_pair_order_and_method_labels() -> None:
    config = ConnectivityConfig(
        enabled=True,
        methods=("coherence_squared", "pli"),
        bands=("alpha",),
        minimum_epochs=2,
    )
    result = compute_connectivity(_epochs(), config, {"alpha": (8.0, 13.0)}, _metadata())
    table = result.table

    assert result.channel_order == ("EEG1", "EEG2", "EEG3")
    assert len(table) == 6
    assert list(
        table[["source_channel", "target_channel"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == [
        ("EEG1", "EEG2"),
        ("EEG1", "EEG3"),
        ("EEG2", "EEG3"),
    ]
    assert set(table["method"]) == {"magnitude_squared_coherence", "pli"}
    assert table["analysis_level"].eq("recording_pooled_epochs").all()
    assert table["frequency_aggregation"].eq("band_average").all()
    coherence = table[table["method"] == "magnitude_squared_coherence"].set_index(
        ["source_channel", "target_channel"]
    )["value"]
    assert coherence[("EEG1", "EEG2")] > 0.9
    assert coherence[("EEG1", "EEG2")] > coherence[("EEG1", "EEG3")]


def test_connectivity_band_selection_and_bad_channel_exclusion() -> None:
    epochs = _epochs(channel_count=4)
    epochs.info["bads"] = ["EEG4"]
    config = ConnectivityConfig(
        enabled=True,
        methods=("coherence_squared",),
        bands=("theta", "alpha"),
        minimum_epochs=2,
    )
    result = compute_connectivity(
        epochs,
        config,
        {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)},
        _metadata(),
    )

    assert result.channel_order == ("EEG1", "EEG2", "EEG3")
    assert len(result.table) == 3 * 2
    assert list(result.table["frequency_band"].unique()) == ["theta", "alpha"]
    assert "EEG4" not in set(result.table["source_channel"]) | set(result.table["target_channel"])


def test_connectivity_requires_channels_and_epochs() -> None:
    config = ConnectivityConfig(
        enabled=True,
        methods=("pli",),
        bands=("alpha",),
        minimum_epochs=3,
    )
    with pytest.raises(ConnectivityError, match="at least 3"):
        compute_connectivity(_epochs(epoch_count=2), config, {"alpha": (8.0, 13.0)}, _metadata())

    one_channel = _epochs(channel_count=2)
    one_channel.info["bads"] = ["EEG2"]
    with pytest.raises(ConnectivityError, match="at least two"):
        compute_connectivity(one_channel, config, {"alpha": (8.0, 13.0)}, _metadata())


def test_connectivity_rejects_nonfinite_input_and_unavailable_band() -> None:
    config = ConnectivityConfig(
        enabled=True,
        methods=("pli",),
        bands=("alpha",),
        minimum_epochs=2,
    )
    nonfinite = _epochs()
    nonfinite._data[0, 0, 0] = np.nan
    with pytest.raises(ConnectivityError, match="nonfinite"):
        compute_connectivity(nonfinite, config, {"alpha": (8.0, 13.0)}, _metadata())

    with pytest.raises(ConnectivityError, match="Nyquist"):
        compute_connectivity(_epochs(), config, {"alpha": (40.0, 55.0)}, _metadata())


def test_connectivity_rejects_nonfinite_estimator_output(monkeypatch) -> None:
    class NonfiniteEstimate:
        def get_data(self):
            return np.full((3, 1), np.nan)

    monkeypatch.setattr(
        "eeg_feature_pipeline.connectivity.spectral_connectivity_epochs",
        lambda *args, **kwargs: NonfiniteEstimate(),
    )
    config = ConnectivityConfig(
        enabled=True,
        methods=("coherence_squared",),
        bands=("alpha",),
        minimum_epochs=2,
    )

    with pytest.raises(ConnectivityError, match="produced nonfinite"):
        compute_connectivity(_epochs(), config, {"alpha": (8.0, 13.0)}, _metadata())
