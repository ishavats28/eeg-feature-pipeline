from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


def config_data(dataset_root: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "root": str(dataset_root),
            "derivatives_root": str(dataset_root.parent / "derivatives"),
            "participant": None,
            "session": None,
            "task": "rest",
            "run": None,
        },
        "preprocessing": {
            "highpass_hz": 1.0,
            "lowpass_hz": 45.0,
            "notch_hz": [50.0],
            "resample_hz": 250.0,
            "reference": "average",
        },
        "epochs": {
            "duration_seconds": 4.0,
            "overlap_seconds": 2.0,
            "reject_by_annotation": True,
        },
        "features": {
            "frequency_bands": {"delta": [1.0, 4.0], "theta": [4.0, 8.0]},
            "selected": ["band_power"],
            "temporal_selected": [
                "rms",
                "variance",
                "hjorth_activity",
                "hjorth_mobility",
                "hjorth_complexity",
            ],
            "analysis_range_hz": [1.0, 8.0],
            "psd_window_seconds": 2.0,
            "psd_overlap_fraction": 0.5,
            "psd_n_fft": None,
        },
        "artifacts": {
            "reject_bad_channels": True,
            "reject_bad_epochs": True,
            "amplitude_threshold_uv": 150.0,
            "flat_threshold_uv": 0.5,
            "noisy_channel_zscore": 6.0,
            "ica_enabled": False,
            "ica_n_components": 0.99,
            "ica_random_state": 97,
            "ica_max_iter": 500,
            "ica_fit_highpass_hz": 1.0,
            "ica_eog_detection": True,
            "ica_eog_threshold": 3.0,
            "ica_min_eeg_channels": 4,
            "ica_min_duration_seconds": 20.0,
        },
        "connectivity": {
            "enabled": False,
            "methods": ["wpli2_debiased"],
            "bands": ["delta", "theta"],
            "minimum_epochs": 2,
            "include_diagonal": False,
        },
        "reliability": {"enabled": True, "segment_count": 2, "method": "spearman"},
        "sensitivity": {"enabled": False, "filter_variants_hz": []},
        "reporting": {"output_formats": ["tsv", "json"], "save_figures": True, "overwrite": False},
    }


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(data: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _write
