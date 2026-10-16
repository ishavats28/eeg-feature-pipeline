"""Self-contained HTML reporting for one completed EEG analysis."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .features import FeatureExtractionResult
from .preprocessing import PreprocessingResult
from .qc import QCResult
from .reliability import StabilityResult
from .sensitivity import SensitivityResult


class ReportingError(RuntimeError):
    """Raised when the local HTML report cannot be generated."""


def _text(value: Any) -> str:
    if value is None:
        return "Not available"
    return html.escape(str(value))


def _number(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{numeric:.{digits}g}" if np.isfinite(numeric) else "Undefined"


def _definition_list(items: list[tuple[str, Any]]) -> str:
    return (
        "<dl>"
        + "".join(f"<dt>{html.escape(label)}</dt><dd>{_text(value)}</dd>" for label, value in items)
        + "</dl>"
    )


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _figure_blocks(figures: dict[str, Path]) -> str:
    captions = {
        "signal": "Independent example EEG trace views before and after preprocessing; sample alignment is not asserted.",
        "psd": "Welch PSD comparison over compatible frequency coverage.",
        "band_power": "Median absolute EEG band power across retained epochs and eligible channels.",
        "epoch_retention": "Initial, rejected, and retained fixed-length epoch counts.",
        "stability": "Median within-recording relative feature differences across disjoint temporal groups.",
        "sensitivity": "Median relative feature changes for matched baseline and alternative preprocessing rows.",
    }
    blocks: list[str] = []
    for name, path in figures.items():
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise ReportingError(f"Could not embed diagnostic figure '{path.name}': {exc}") from exc
        blocks.append(
            "<figure>"
            f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(name)} diagnostic figure'>"
            f"<figcaption>{html.escape(captions.get(name, name.replace('_', ' ').title()))}</figcaption>"
            "</figure>"
        )
    return "".join(blocks) or "<p>No diagnostic figures were requested or available.</p>"


def _spectral_summary(features: FeatureExtractionResult) -> str:
    table = features.spectral_table
    rows: list[list[Any]] = []
    for band, frame in table[table["frequency_band"] != "analysis_range"].groupby(
        "frequency_band", sort=False
    ):
        rows.append(
            [
                band,
                _number(frame["absolute_power_uv2"].median()),
                _number(frame["relative_power"].median()),
                len(frame),
            ]
        )
    entropy = table.loc[table["frequency_band"] == "analysis_range", "spectral_entropy"].median()
    return (
        _table(
            ["Band", "Median absolute power (µV²)", "Median relative power", "Rows"],
            rows,
        )
        + f"<p>Median normalized spectral entropy: {_number(entropy)}.</p>"
    )


def _temporal_summary(features: FeatureExtractionResult) -> str:
    table = features.temporal_table
    rows = [
        ["RMS", _number(table["rms_uv"].median()), "µV"],
        ["Variance", _number(table["variance_uv2"].median()), "µV²"],
        ["Hjorth activity", _number(table["hjorth_activity_uv2"].median()), "µV²"],
        ["Hjorth mobility", _number(table["hjorth_mobility"].median()), "dimensionless"],
        ["Hjorth complexity", _number(table["hjorth_complexity"].median()), "dimensionless"],
    ]
    return _table(["Feature", "Median", "Unit"], rows)


def generate_html_report(
    output_path: str | Path,
    *,
    recording: Any,
    config: PipelineConfig,
    preprocessing: PreprocessingResult,
    features: FeatureExtractionResult,
    qc: QCResult | None,
    stability: StabilityResult | None,
    sensitivity: SensitivityResult | None,
    figures: dict[str, Path],
    stages: dict[str, dict[str, Any]],
    output_references: dict[str, str],
    provenance: dict[str, Any],
) -> Path:
    """Write an offline HTML report containing actual computed summaries."""
    destination = Path(output_path).expanduser().resolve()
    rejected = preprocessing.initial_epoch_count - preprocessing.retained_epoch_count
    preprocessing_section = _definition_list(
        [
            ("Original sampling frequency", f"{preprocessing.original_sampling_frequency_hz:g} Hz"),
            ("Final sampling frequency", f"{preprocessing.final_sampling_frequency_hz:g} Hz"),
            ("Original EEG channels", preprocessing.original_eeg_channel_count),
            ("Retained eligible EEG channels", preprocessing.retained_eeg_channel_count),
            ("Existing bad channels", ", ".join(preprocessing.existing_bad_channels) or "None"),
            (
                "Newly detected bad channels",
                ", ".join(preprocessing.detected_bad_channels) or "None",
            ),
            ("Reference", preprocessing.parameters.get("reference")),
            (
                "Passband",
                f"{preprocessing.parameters.get('highpass_hz')}–{preprocessing.parameters.get('lowpass_hz')} Hz",
            ),
            ("ICA status", preprocessing.ica_status),
            ("ICA excluded components", list(preprocessing.ica_removed_components) or "None"),
            (
                "Initial / rejected / retained epochs",
                f"{preprocessing.initial_epoch_count} / {rejected} / {preprocessing.retained_epoch_count}",
            ),
        ]
    )
    rejection_rows = [
        [reason, count] for reason, count in sorted(preprocessing.epoch_rejection_reasons.items())
    ]
    if rejection_rows:
        preprocessing_section += _table(["Epoch rejection reason", "Count"], rejection_rows)

    connectivity_stage = stages["connectivity"]
    if connectivity_stage["status"] == "completed" and not features.connectivity_table.empty:
        connectivity_rows: list[list[Any]] = []
        for method, frame in features.connectivity_table.groupby("method", sort=True):
            connectivity_rows.append(
                [
                    method,
                    len(frame),
                    _number(frame["value"].min()),
                    _number(frame["value"].median()),
                    _number(frame["value"].max()),
                ]
            )
        connectivity_html = _table(
            ["Method", "Rows", "Minimum", "Median", "Maximum"], connectivity_rows
        )
        connectivity_html += (
            "<p>Estimates pool retained epochs at the recording level and are band averaged. "
            "They are sensor-space statistical dependencies, not anatomical or source-level connections. "
            "Volume conduction and reference choice limit interpretation.</p>"
        )
    else:
        connectivity_html = f"<p>Status: {_text(connectivity_stage['status'])}. {_text(connectivity_stage.get('reason'))}</p>"

    if stability is not None:
        stability_rows = [
            [
                item["feature_family"],
                item["feature"],
                item["matched_observations"],
                _number(item["median_absolute_difference"]),
                _number(item["median_relative_difference"]),
            ]
            for item in stability.summary["descriptive_summaries"]
        ]
        stability_html = _definition_list(
            [
                ("Status", stability.summary["status"]),
                ("Group 1 epochs", stability.summary["group_1_epoch_count"]),
                ("Group 2 epochs", stability.summary["group_2_epoch_count"]),
                ("Matched features", stability.summary.get("matched_feature_count")),
                ("Boundary epochs omitted", stability.summary["boundary_epochs_omitted"]),
            ]
        )
        if stability_rows:
            stability_html += _table(
                [
                    "Family",
                    "Feature",
                    "Matched",
                    "Median absolute difference",
                    "Median relative difference",
                ],
                stability_rows,
            )
        stability_html += (
            "<p>This is within-recording feature stability across temporally disjoint groups, "
            "not intersession test–retest reliability or ICC. Overlapping epochs within each group remain dependent.</p>"
        )
    else:
        stability_html = f"<p>Status: {_text(stages['stability']['status'])}. {_text(stages['stability'].get('reason'))}</p>"

    if sensitivity is not None:
        sensitivity_rows = [
            [
                item["feature_family"],
                item["feature"],
                item["matched_rows"],
                _number(item["median_absolute_change"]),
                _number(item["median_relative_change"]),
            ]
            for item in sensitivity.comparison["descriptive_summaries"]
        ]
        sensitivity_html = _definition_list(
            [
                ("Status", sensitivity.comparison["status"]),
                ("Comparison scope", sensitivity.comparison["comparison_scope"]),
                ("Matched feature rows", sensitivity.comparison["matched_feature_rows"]),
                (
                    "Parameter differences",
                    json.dumps(sensitivity.comparison["parameter_differences"], sort_keys=True),
                ),
            ]
        )
        if sensitivity_rows:
            sensitivity_html += _table(
                [
                    "Family",
                    "Feature",
                    "Matched rows",
                    "Median absolute change",
                    "Median relative change",
                ],
                sensitivity_rows,
            )
        sensitivity_html += "<p>This descriptive comparison does not measure biological accuracy. Differing retained data can confound changes.</p>"
    else:
        sensitivity_html = f"<p>Status: {_text(stages['sensitivity']['status'])}. {_text(stages['sensitivity'].get('reason'))}</p>"

    warning_values = list(preprocessing.warnings)
    if qc is not None:
        warning_values.extend(qc.metrics.get("warnings", []))
    warning_values.extend(provenance.get("warnings", []))
    warnings_html = (
        "<ul>"
        + "".join(f"<li>{_text(value)}</li>" for value in dict.fromkeys(warning_values))
        + "</ul>"
        if warning_values
        else "<p>No processing warnings were recorded.</p>"
    )
    output_rows = [[name, value] for name, value in sorted(output_references.items())]
    version_rows = [[name, version] for name, version in provenance["software_versions"].items()]
    bands = ", ".join(
        f"{name} ({low:g}–{high:g} Hz)"
        for name, (low, high) in config.features.frequency_bands.items()
    )
    css = """
    :root { color-scheme: light; --ink:#17212b; --muted:#52606d; --line:#d8e0e8; --accent:#245b78; }
    body { margin:0; background:#f4f7f9; color:var(--ink); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }
    main { max-width:1050px; margin:0 auto; padding:32px 24px 64px; }
    header, section { background:white; border:1px solid var(--line); border-radius:10px; padding:22px 26px; margin:0 0 18px; }
    h1 { margin:0 0 6px; color:var(--accent); } h2 { color:var(--accent); margin-top:0; }
    .subtitle, .note { color:var(--muted); } dl { display:grid; grid-template-columns:minmax(190px, 1fr) 2fr; gap:6px 18px; }
    dt { font-weight:650; } dd { margin:0; overflow-wrap:anywhere; }
    .table-wrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; margin:12px 0; }
    th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; } th { background:#eef4f7; }
    figure { margin:20px 0 28px; } img { display:block; max-width:100%; height:auto; border:1px solid var(--line); }
    figcaption { color:var(--muted); margin-top:7px; } code { background:#eef2f5; padding:2px 5px; border-radius:4px; }
    @media (max-width:650px) { dl { grid-template-columns:1fr; } dd { margin-bottom:8px; } }
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG feature pipeline report: {html.escape(recording.label)}</title><style>{
        css
    }</style></head>
<body><main><header><h1>EEG Feature Pipeline Report</h1><p class="subtitle">Recording {
        _text(recording.label)
    }</p>
<p class="subtitle">Author: Isha Vats</p>
<p class="note">Generated from computed outputs. No real-data validation claim is implied.</p></header>
<section><h2>Dataset and recording</h2>{
        _definition_list(
            [
                ("Subject", recording.participant),
                ("Session", recording.session),
                ("Task", recording.task),
                ("Acquisition", recording.acquisition),
                ("Run", recording.run),
                ("Input file", recording.label),
                ("Epoch duration", f"{config.epochs.duration_seconds:g} s"),
                ("Epoch overlap", f"{config.epochs.overlap_seconds:g} s"),
            ]
        )
    }</section>
<section><h2>Preprocessing</h2>{preprocessing_section}<h3>Warnings and limitations</h3>{
        warnings_html
    }</section>
<section><h2>Feature extraction</h2><p>Configured bands: {_text(bands)}.</p>
<p>Absolute power is Welch PSD integrated by the trapezoidal rule in µV². Relative power uses the full analysis-range power denominator. Spectral entropy is normalized base-2 Shannon entropy.</p>
{
        _spectral_summary(features)
    }<p>Temporal definitions use population variance and discrete first differences.</p>{
        _temporal_summary(features)
    }</section>
<section><h2>Functional connectivity</h2>{connectivity_html}</section>
<section><h2>Within-recording feature stability</h2>{stability_html}</section>
<section><h2>Preprocessing sensitivity</h2>{sensitivity_html}</section>
<section><h2>Diagnostic figures</h2>{_figure_blocks(figures)}</section>
<section><h2>Outputs</h2>{_table(["Output", "Relative path"], output_rows)}
<p class="note">Custom TSV, JSON, HTML, PNG, and MNE Epochs FIF files are pipeline-specific outputs and are not claimed to be fully standardized BIDS derivative data types.</p></section>
<section><h2>Reproducibility</h2>{
        _definition_list(
            [
                ("Pipeline version", provenance["pipeline_version"]),
                ("Processing timestamp", provenance["processing_timestamp"]),
                ("Configuration snapshot", provenance["configuration_snapshot"]),
                ("ICA random state", provenance["random_seeds"]["ica"]),
            ]
        )
    }
{_table(["Software", "Version"], version_rows)}</section>
</main></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(document, encoding="utf-8")
    except OSError as exc:
        raise ReportingError(f"Could not write HTML report: {exc}") from exc
    return destination
