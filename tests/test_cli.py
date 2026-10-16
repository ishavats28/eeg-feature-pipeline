from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import config_data

from eeg_feature_pipeline import __version__, cli


def test_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "list-recordings" in output
    assert "inspect-recording" in output
    assert "preprocess" in output
    assert "extract-features" in output
    assert "run" in output
    assert "analyze-stability" in output
    assert "analyze-sensitivity" in output
    assert "qc" in output


def test_version(capsys) -> None:
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_validate_config(tmp_path: Path, write_config, capsys) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    path = write_config(config_data(dataset))

    assert cli.main(["validate-config", "--config", str(path)]) == 0
    assert "Configuration is valid" in capsys.readouterr().out


def test_validate_config_failure(tmp_path: Path, write_config, capsys) -> None:
    path = write_config(config_data(tmp_path / "missing"))

    assert cli.main(["validate-config", "--config", str(path)]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_list_recordings_uses_config_selectors(
    tmp_path: Path, write_config, monkeypatch, capsys
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    path = write_config(config_data(dataset))
    seen: dict[str, object] = {}

    def discover(root, **selectors):
        seen.update(selectors)
        return [SimpleNamespace(label="sub-01/eeg/sub-01_task-rest_eeg.vhdr")]

    monkeypatch.setattr(cli, "discover_recordings", discover)
    assert cli.main(["list-recordings", "--config", str(path)]) == 0
    assert seen["task"] == "rest"
    assert "Found 1 recording(s)." in capsys.readouterr().out


def test_inspect_selector_override(tmp_path: Path, write_config, monkeypatch, capsys) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    path = write_config(config_data(dataset))
    seen: dict[str, object] = {}

    def select(root, **selectors):
        seen.update(selectors)
        return SimpleNamespace(label="recording")

    monkeypatch.setattr(cli, "select_recording", select)
    monkeypatch.setattr(cli, "inspect_recording", lambda recording: {"n_channels": 2})

    status = cli.main(
        ["inspect-recording", "--config", str(path), "--participant", "sub-02", "--run", "03"]
    )

    assert status == 0
    assert seen["participant"] == "02"
    assert seen["run"] == "03"
    assert '"n_channels": 2' in capsys.readouterr().out


def test_preprocess_success_with_subject_alias(
    tmp_path: Path, write_config, monkeypatch, capsys
) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    path = write_config(config_data(dataset))
    seen: dict[str, object] = {}

    def select(root, **selectors):
        seen.update(selectors)
        return SimpleNamespace(label="recording")

    result = SimpleNamespace(
        retained_epoch_count=3,
        summary=lambda: {"epochs": {"initial": 4, "retained": 3}},
    )
    monkeypatch.setattr(cli, "select_recording", select)
    monkeypatch.setattr(cli, "load_recording", lambda recording: SimpleNamespace())
    monkeypatch.setattr(cli, "preprocess_raw", lambda raw, config: result)

    status = cli.main(["preprocess", "--config", str(path), "--subject", "sub-02", "--run", "01"])

    assert status == 0
    assert seen["participant"] == "02"
    assert seen["run"] == "01"
    assert '"retained": 3' in capsys.readouterr().out


def test_preprocess_failure_is_nonzero(tmp_path: Path, write_config, monkeypatch, capsys) -> None:
    dataset = tmp_path / "bids"
    dataset.mkdir()
    path = write_config(config_data(dataset))
    monkeypatch.setattr(cli, "select_recording", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, "load_recording", lambda recording: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "preprocess_raw",
        lambda raw, config: (_ for _ in ()).throw(cli.PreprocessingError("no usable EEG")),
    )

    assert cli.main(["preprocess", "--config", str(path), "--subject", "01"]) == 2
    assert "no usable EEG" in capsys.readouterr().err


def test_preprocess_cli_with_synthetic_bids(tmp_path: Path, write_config, capsys) -> None:
    mne = pytest.importorskip("mne")
    mne_bids = pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    root = tmp_path / "synthetic_bids"
    sfreq = 200.0
    times = np.arange(int(6 * sfreq)) / sfreq
    data = np.vstack(
        [5e-6 * np.sin(2 * np.pi * frequency * times) for frequency in (8, 10, 12, 14)]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    raw.info["line_freq"] = 50.0
    bids_path = mne_bids.BIDSPath(subject="01", task="rest", run="01", datatype="eeg", root=root)
    mne_bids.write_raw_bids(
        raw,
        bids_path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose="ERROR",
    )
    data_config = config_data(root)
    data_config["dataset"].update(participant="01", run="01")
    data_config["preprocessing"].update(notch_hz=[], resample_hz=100.0)
    data_config["epochs"].update(duration_seconds=2.0, overlap_seconds=1.0)
    data_config["artifacts"].update(
        reject_bad_channels=False,
        reject_bad_epochs=False,
        ica_enabled=False,
    )
    config_path = write_config(data_config, "synthetic-config.yaml")

    status = cli.main(["preprocess", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert status == 0
    assert '"status": "disabled"' in captured.out
    assert '"retained": 5' in captured.out

    feature_output = tmp_path / "feature-output"
    feature_status = cli.main(
        [
            "extract-features",
            "--config",
            str(config_path),
            "--output-dir",
            str(feature_output),
        ]
    )
    feature_capture = capsys.readouterr()
    assert feature_status == 0
    assert '"spectral_rows": 40' in feature_capture.out
    assert '"temporal_rows": 20' in feature_capture.out
    assert len(list(feature_output.glob("*_features.tsv"))) == 2

    stability_output = tmp_path / "stability-output"
    stability_status = cli.main(
        [
            "analyze-stability",
            "--config",
            str(config_path),
            "--output-dir",
            str(stability_output),
        ]
    )
    stability_capture = capsys.readouterr()
    assert stability_status == 0
    assert '"analysis": "within-recording feature stability"' in stability_capture.out
    assert list(stability_output.glob("*_desc-stability_features.tsv"))
    assert list(stability_output.glob("*_desc-stability_summary.json"))

    qc_output = tmp_path / "qc-output"
    qc_status = cli.main(["qc", "--config", str(config_path), "--output-dir", str(qc_output)])
    qc_capture = capsys.readouterr()
    assert qc_status == 0
    assert '"rejection_fraction": 0.0' in qc_capture.out
    assert list(qc_output.glob("*_desc-qc_metrics.json"))
    assert list(qc_output.glob("*.png"))

    data_config["sensitivity"].update(
        enabled=True,
        filter_variants_hz=[[0.5, 45.0]],
    )
    sensitivity_config = write_config(data_config, "sensitivity-config.yaml")
    sensitivity_output = tmp_path / "sensitivity-output"
    sensitivity_status = cli.main(
        [
            "analyze-sensitivity",
            "--config",
            str(sensitivity_config),
            "--output-dir",
            str(sensitivity_output),
        ]
    )
    sensitivity_capture = capsys.readouterr()
    assert sensitivity_status == 0
    assert '"analysis": "preprocessing sensitivity"' in sensitivity_capture.out
    assert list(sensitivity_output.glob("*_desc-sensitivity_features.tsv"))
    assert list(sensitivity_output.glob("*_desc-sensitivity_config.json"))
