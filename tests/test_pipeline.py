from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import config_data

from eeg_feature_pipeline import cli
from eeg_feature_pipeline.bids import BIDSSelectionError, Recording
from eeg_feature_pipeline.config import load_config
from eeg_feature_pipeline.connectivity import ConnectivityError
from eeg_feature_pipeline.derivatives import DerivativeError, build_layout
from eeg_feature_pipeline.pipeline import run_pipeline


def _synthetic_bids(root: Path, *, duration: float = 10.0) -> Path:
    mne = pytest.importorskip("mne")
    mne_bids = pytest.importorskip("mne_bids")
    pytest.importorskip("pybv")
    sfreq = 200.0
    times = np.arange(int(duration * sfreq)) / sfreq
    rng = np.random.default_rng(2025)
    common = 6e-6 * np.sin(2 * np.pi * 10 * times)
    data = np.vstack(
        [
            common
            + 2e-6 * np.sin(2 * np.pi * frequency * times)
            + rng.normal(0, 0.08e-6, len(times))
            for frequency in (6, 8, 12, 16)
        ]
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    raw.info["line_freq"] = 50.0
    path = mne_bids.BIDSPath(subject="01", task="rest", run="01", datatype="eeg", root=root)
    mne_bids.write_raw_bids(
        raw,
        path,
        format="BrainVision",
        allow_preload=True,
        overwrite=True,
        verbose="ERROR",
    )
    return Path(path.copy().update(extension=".vhdr").fpath)


def _pipeline_config(
    tmp_path: Path,
    write_config,
    *,
    connectivity: bool = False,
    stability: bool = True,
    sensitivity: bool = False,
    figures: bool = True,
    qc: bool = True,
    minimum_connectivity_epochs: int = 2,
):
    root = tmp_path / "bids"
    _synthetic_bids(root)
    data = config_data(root)
    data["dataset"].update(participant="01", run="01")
    data["preprocessing"].update(notch_hz=[], resample_hz=100.0)
    data["epochs"].update(duration_seconds=2.0, overlap_seconds=1.0)
    data["artifacts"].update(reject_bad_channels=False, reject_bad_epochs=False)
    data["connectivity"].update(
        enabled=connectivity,
        methods=["pli"],
        bands=["delta", "theta"],
        minimum_epochs=minimum_connectivity_epochs,
    )
    data["reliability"].update(enabled=stability, minimum_epochs_per_group=2)
    data["sensitivity"].update(
        enabled=sensitivity,
        filter_variants_hz=[[0.5, 45.0]],
    )
    data["reporting"].update(save_figures=figures, qc_enabled=qc, figure_dpi=72)
    return load_config(write_config(data, "pipeline-config.yaml")), root


def test_full_unified_run_outputs_manifest_report_and_preserves_source(
    tmp_path: Path, write_config, monkeypatch
) -> None:
    config, source_root = _pipeline_config(
        tmp_path,
        write_config,
        connectivity=True,
        sensitivity=True,
    )
    source_file = next(source_root.rglob("*.eeg"))
    source_before = source_file.read_bytes()
    output = tmp_path / "unified-output"
    import eeg_feature_pipeline.pipeline as pipeline_module

    original_loader = pipeline_module.load_recording
    load_count = 0

    def counted_loader(recording):
        nonlocal load_count
        load_count += 1
        return original_loader(recording)

    monkeypatch.setattr(pipeline_module, "load_recording", counted_loader)

    result = run_pipeline(config, output_dir=output)

    assert load_count == 1
    assert result.derivative_root == output.resolve()
    assert result.stages["preprocessing"]["status"] == "completed"
    assert result.stages["connectivity"]["status"] == "completed"
    assert result.stages["stability"]["status"] == "ok"
    assert result.stages["sensitivity"]["status"] == "ok"
    assert result.stages["qc"]["status"] == "completed"
    assert source_file.read_bytes() == source_before

    description = json.loads((output / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert description["Authors"] == ["Isha Vats"]
    assert description["GeneratedBy"][0] == {
        "Description": (
            "Resting-state EEG preprocessing, feature extraction, stability, "
            "sensitivity, and quality-control outputs."
        ),
        "Name": "eeg-feature-pipeline",
        "Version": "1.0.0",
    }
    recording_dir = output / "sub-01" / "eeg"
    assert recording_dir.is_dir()
    assert all(path.exists() and path.stat().st_size > 0 for path in result.outputs.values())
    assert "sub-01_task-rest_run-01" in result.outputs["report"].name

    spectral = pd.read_csv(result.outputs["spectral_features"], sep="\t")
    temporal = pd.read_csv(result.outputs["temporal_features"], sep="\t")
    connectivity = pd.read_csv(result.outputs["connectivity_features"], sep="\t")
    assert not spectral.empty
    assert not temporal.empty
    assert not connectivity.empty
    assert set(connectivity["analysis_level"]) == {"recording_pooled_epochs"}

    manifest = json.loads(result.outputs["execution_manifest"].read_text())
    assert manifest["status"] == "completed"
    assert manifest["input_recording"].endswith("_eeg.vhdr")
    assert manifest["enabled_stages"]["sensitivity"] is True
    assert manifest["random_seeds"]["ica"] == 97
    assert all(not Path(value).is_absolute() for value in manifest["outputs"].values())
    assert all((output / value).exists() for value in manifest["outputs"].values())
    assert any("Connectivity band 'delta'" in warning for warning in manifest["warnings"])

    snapshot = json.loads(result.outputs["processing_config"].read_text())
    assert snapshot["effective_configuration"]["dataset"]["root"] == "<source-bids-root>"
    assert str(source_root.resolve()) not in result.outputs["processing_config"].read_text()

    report = result.outputs["report"].read_text(encoding="utf-8")
    expected_power = spectral.loc[
        spectral["frequency_band"] == "delta", "absolute_power_uv2"
    ].median()
    assert "EEG Feature Pipeline Report" in report
    assert "Author: Isha Vats" in report
    assert f"{expected_power:.4g}" in report
    assert "Within-recording feature stability" in report
    assert "not intersession test–retest reliability" in report
    assert "recording level" in report
    assert "data:image/png;base64," in report
    assert "<script" not in report
    assert "https://" not in report and "http://" not in report
    assert result.outputs["spectral_features"].name in report
    assert "Connectivity band &#x27;delta&#x27;" in report
    HTMLParser().feed(report)


def test_disabled_optional_stages_and_configured_derivative_parent(
    tmp_path: Path, write_config
) -> None:
    config, _ = _pipeline_config(
        tmp_path,
        write_config,
        connectivity=False,
        stability=False,
        sensitivity=False,
        figures=False,
        qc=False,
    )

    result = run_pipeline(config)

    assert result.derivative_root == (config.dataset.derivatives_root / "eeg-feature-pipeline")
    for stage in ("connectivity", "stability", "sensitivity", "qc", "figures"):
        assert result.stages[stage]["status"] == "skipped"
    assert "connectivity_features" not in result.outputs
    assert "qc_metrics" not in result.outputs
    assert not list(result.derivative_root.rglob("*.png"))
    assert result.outputs["report"].is_file()


def test_collision_preflight_and_explicit_overwrite(
    tmp_path: Path, write_config, monkeypatch
) -> None:
    config, _ = _pipeline_config(tmp_path, write_config, figures=False)
    output = tmp_path / "collision-output"
    first = run_pipeline(config, output_dir=output)
    unrelated = output / "sub-01" / "eeg" / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    def must_not_load(recording):
        raise AssertionError("collision check should precede source loading")

    import eeg_feature_pipeline.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "load_recording", must_not_load)
    with pytest.raises(DerivativeError, match="overwrite is disabled"):
        run_pipeline(config, output_dir=output)
    monkeypatch.undo()

    second = run_pipeline(config, output_dir=output, overwrite=True)
    assert second.outputs["execution_manifest"].is_file()
    assert first.outputs["report"] == second.outputs["report"]
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_required_connectivity_failure_leaves_no_completion_manifest(
    tmp_path: Path, write_config
) -> None:
    config, _ = _pipeline_config(
        tmp_path,
        write_config,
        connectivity=True,
        minimum_connectivity_epochs=20,
    )
    output = tmp_path / "failed-output"

    with pytest.raises(ConnectivityError, match="at least 20"):
        run_pipeline(config, output_dir=output)

    assert not list(output.rglob("*_desc-execution_manifest.json"))
    assert not list(output.rglob("*.tsv"))
    assert not list(output.glob(".*.staging-*"))


def test_run_missing_selection_and_entity_layout(tmp_path: Path, write_config) -> None:
    config, _ = _pipeline_config(tmp_path, write_config)
    with pytest.raises(BIDSSelectionError, match="No BIDS EEG recording"):
        run_pipeline(config, participant="99", output_dir=tmp_path / "none")

    recording = Recording(
        path=None,
        relative_path=Path("sub-01/ses-a/eeg/source.vhdr"),
        participant="01",
        session="a",
        task="rest",
        run="02",
        acquisition="high",
        recording="main",
        extension=".vhdr",
    )
    layout = build_layout(tmp_path / "derivatives", recording)
    assert layout.recording_dir.parts[-2:] == ("ses-a", "eeg")
    assert layout.stem == "sub-01_ses-a_task-rest_acq-high_run-02_recording-main"


def test_run_cli_human_and_json_summaries(tmp_path: Path, write_config, capsys) -> None:
    config, _ = _pipeline_config(tmp_path, write_config, figures=False)
    output = tmp_path / "cli-output"

    status = cli.main(
        [
            "run",
            "--config",
            str(config.source_path),
            "--subject",
            "01",
            "--output-dir",
            str(output),
        ]
    )
    human = capsys.readouterr().out
    assert status == 0
    assert "Completed:" in human
    assert "Report:" in human
    assert "connectivity=skipped" in human

    json_status = cli.main(
        [
            "run",
            "--config",
            str(config.source_path),
            "--output-dir",
            str(output),
            "--overwrite",
            "--json",
        ]
    )
    parsed = json.loads(capsys.readouterr().out)
    assert json_status == 0
    assert parsed["status"] == "completed"
    assert parsed["outputs"]["report"].endswith("_desc-report.html")
