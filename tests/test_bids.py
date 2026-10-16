from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_feature_pipeline import bids
from eeg_feature_pipeline.bids import BIDSError, BIDSSelectionError


def _description(root: Path) -> None:
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Synthetic test dataset", "BIDSVersion": "1.8.0"}),
        encoding="utf-8",
    )


def test_missing_dataset_description_fails(tmp_path: Path) -> None:
    with pytest.raises(BIDSError, match="missing required file"):
        bids.validate_dataset(tmp_path)


def test_malformed_dataset_description_fails(tmp_path: Path) -> None:
    (tmp_path / "dataset_description.json").write_text("{bad json", encoding="utf-8")
    with pytest.raises(BIDSError, match="valid JSON"):
        bids.validate_dataset(tmp_path)


def test_missing_required_description_field_fails(tmp_path: Path) -> None:
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "Incomplete"}), encoding="utf-8"
    )
    with pytest.raises(BIDSError, match="BIDSVersion"):
        bids.validate_dataset(tmp_path)


def test_discovery_passes_selectors_without_loading(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bids"
    _description(root)
    data_file = root / "sub-01" / "eeg" / "sub-01_task-rest_run-01_eeg.vhdr"
    data_file.parent.mkdir(parents=True)
    data_file.touch()
    path = SimpleNamespace(
        fpath=data_file,
        subject="01",
        session=None,
        task="rest",
        run="01",
        acquisition=None,
        recording=None,
        extension=".vhdr",
    )
    calls: dict[str, object] = {}

    def find_matching_paths(**kwargs):
        calls.update(kwargs)
        return [path]

    monkeypatch.setattr(
        bids, "_mne_bids", lambda: SimpleNamespace(find_matching_paths=find_matching_paths)
    )

    records = bids.discover_recordings(root, participant="01", task="rest", run="01")

    assert [record.label for record in records] == [data_file.relative_to(root).as_posix()]
    assert calls["subjects"] == "01"
    assert calls["tasks"] == "rest"
    assert calls["ignore_json"] is True


def test_ambiguous_selection_is_explicit(tmp_path: Path, monkeypatch) -> None:
    records = [SimpleNamespace(label="first.vhdr"), SimpleNamespace(label="second.vhdr")]
    monkeypatch.setattr(bids, "discover_recordings", lambda *args, **kwargs: records)

    with pytest.raises(BIDSSelectionError, match="ambiguous: 2"):
        bids.select_recording(tmp_path, participant="01")


def test_no_matching_selection_is_explicit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bids, "discover_recordings", lambda *args, **kwargs: [])

    with pytest.raises(BIDSSelectionError, match="No BIDS EEG recording"):
        bids.select_recording(tmp_path, participant="99")


def test_nonconforming_eeg_path_is_reported(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bids"
    _description(root)
    malformed = SimpleNamespace(fpath=root / "sub-01_bad_eeg.edf", check=False)

    def find_matching_paths(*, check, **kwargs):
        return [] if check else [malformed]

    monkeypatch.setattr(
        bids,
        "_mne_bids",
        lambda: SimpleNamespace(find_matching_paths=find_matching_paths),
    )

    with pytest.raises(BIDSError, match="non-conforming BIDS paths"):
        bids.discover_recordings(root)


def test_synthetic_bids_discovery_and_inspection(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    mne_bids = pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    root = tmp_path / "synthetic_bids"
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types="eeg")
    samples = np.zeros((2, 500), dtype=float)
    samples[0] = 1e-6 * np.sin(2 * np.pi * 10 * np.arange(500) / 100.0)
    raw = mne.io.RawArray(samples, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=[1.0], duration=[0.25], description=["test-marker"]))
    path = mne_bids.BIDSPath(subject="01", task="rest", run="01", datatype="eeg", root=root)
    mne_bids.write_raw_bids(
        raw,
        path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose="ERROR",
    )

    records = bids.discover_recordings(root, participant="01", task="rest", run="01")
    metadata = bids.inspect_recording(records[0])

    assert len(records) == 1
    assert metadata["sampling_frequency_hz"] == 100.0
    assert metadata["n_channels"] == 2
    assert metadata["channel_types"] == {"eeg": 2}
    assert metadata["annotation_count"] == 1
    assert metadata["annotations"] == ["test-marker"]
