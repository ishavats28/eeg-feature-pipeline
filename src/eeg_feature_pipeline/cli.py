"""Command-line interface for configuration and BIDS inspection."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace

from . import __version__
from .bids import (
    BIDSError,
    discover_recordings,
    inspect_recording,
    load_recording,
    select_recording,
)
from .config import ConfigError, PipelineConfig, load_config
from .connectivity import ConnectivityError
from .derivatives import DerivativeError
from .features import FeatureExtractionError, extract_features, save_feature_tables
from .pipeline import PipelineExecutionError, run_pipeline
from .plots import PlotError, generate_diagnostic_figures
from .preprocessing import PreprocessingError, preprocess_raw, save_preprocessing_result
from .qc import QCError, collect_qc_metrics, save_qc_metrics
from .reliability import (
    ReliabilityError,
    analyze_within_recording_stability,
    save_stability_results,
)
from .reporting import ReportingError
from .sensitivity import (
    SensitivityError,
    analyze_preprocessing_sensitivity,
    save_sensitivity_results,
)


def _entity_argument(name: str):
    prefixes = {"participant": "sub-", "session": "ses-", "task": "task-", "run": "run-"}

    def parse(value: str) -> str:
        normalized = value.removeprefix(prefixes[name]).strip()
        if not normalized or any(character in normalized for character in "/\\_"):
            raise argparse.ArgumentTypeError(
                f"{name} must be a BIDS entity value without slashes or underscores"
            )
        return normalized

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eeg-feature-pipeline",
        description="Validate, inspect, preprocess, and extract BIDS EEG features.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the package version.")

    validate = subparsers.add_parser("validate-config", help="Validate a YAML configuration.")
    validate.add_argument("--config", required=True, help="Path to the YAML configuration file.")

    listing = subparsers.add_parser("list-recordings", help="List matching BIDS EEG recordings.")
    listing.add_argument("--config", required=True, help="Path to the YAML configuration file.")

    inspect = subparsers.add_parser("inspect-recording", help="Inspect one BIDS EEG recording.")
    inspect.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    for flag in ("participant", "session", "task", "run"):
        inspect.add_argument(
            f"--{flag}",
            type=_entity_argument(flag),
            help=f"Override the configured {flag} selector.",
        )

    preprocess = subparsers.add_parser(
        "preprocess", help="Preprocess and epoch one BIDS EEG recording."
    )
    preprocess.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    preprocess.add_argument(
        "--participant",
        "--subject",
        dest="participant",
        type=_entity_argument("participant"),
        help="Override the configured participant selector.",
    )
    for flag in ("session", "task", "run"):
        preprocess.add_argument(
            f"--{flag}",
            type=_entity_argument(flag),
            help=f"Override the configured {flag} selector.",
        )
    preprocess.add_argument(
        "--save",
        action="store_true",
        help="Save cleaned Raw and Epochs FIF files to a local processing-output directory.",
    )
    preprocess.add_argument(
        "--output-dir",
        help="Output directory used with --save (default: derivatives_root/preprocessing).",
    )
    preprocess.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing preprocessing FIF outputs.",
    )

    extract = subparsers.add_parser(
        "extract-features",
        help="Preprocess one recording and save spectral, temporal, and connectivity features.",
    )
    extract.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    extract.add_argument(
        "--participant",
        "--subject",
        dest="participant",
        type=_entity_argument("participant"),
        help="Override the configured participant selector.",
    )
    for flag in ("session", "task", "run"):
        extract.add_argument(
            f"--{flag}",
            type=_entity_argument(flag),
            help=f"Override the configured {flag} selector.",
        )
    extract.add_argument(
        "--output-dir",
        help="Feature output directory (default: derivatives_root/features).",
    )
    extract.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing feature TSV files.",
    )

    run_command = subparsers.add_parser(
        "run", help="Run the configured single-recording analysis and HTML report."
    )
    run_command.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    run_command.add_argument(
        "--participant",
        "--subject",
        dest="participant",
        type=_entity_argument("participant"),
        help="Override the configured participant selector.",
    )
    for flag in ("session", "task", "run"):
        run_command.add_argument(
            f"--{flag}",
            type=_entity_argument(flag),
            help=f"Override the configured {flag} selector.",
        )
    run_command.add_argument(
        "--output-dir",
        help="Explicit pipeline derivative root (default: configured derivatives root).",
    )
    run_command.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this recording's known pipeline outputs.",
    )
    run_command.add_argument(
        "--json",
        action="store_true",
        help="Print the execution summary as JSON instead of readable text.",
    )

    for command, help_text in (
        ("analyze-stability", "Analyze within-recording feature stability."),
        ("analyze-sensitivity", "Compare baseline and alternative preprocessing."),
        ("qc", "Save preprocessing QC metrics and diagnostic figures."),
    ):
        analysis = subparsers.add_parser(command, help=help_text)
        analysis.add_argument(
            "--config", required=True, help="Path to the YAML configuration file."
        )
        analysis.add_argument(
            "--participant",
            "--subject",
            dest="participant",
            type=_entity_argument("participant"),
            help="Override the configured participant selector.",
        )
        for flag in ("session", "task", "run"):
            analysis.add_argument(
                f"--{flag}",
                type=_entity_argument(flag),
                help=f"Override the configured {flag} selector.",
            )
        analysis.add_argument(
            "--output-dir",
            help="Output directory (default: derivatives_root/quality-control).",
        )
        analysis.add_argument(
            "--overwrite",
            action="store_true",
            help="Allow replacement of existing analysis outputs.",
        )
    return parser


def _selectors(
    config: PipelineConfig, args: argparse.Namespace | None = None
) -> dict[str, str | None]:
    result = {
        "participant": config.dataset.participant,
        "session": config.dataset.session,
        "task": config.dataset.task,
        "run": config.dataset.run,
    }
    if args is not None:
        for name in result:
            value = getattr(args, name, None)
            if value is not None:
                result[name] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "version":
            print(__version__)
            return 0
        config = load_config(args.config)
        if args.command == "validate-config":
            print(f"Configuration is valid: {config.source_path}")
            print(f"BIDS dataset root: {config.dataset.root}")
            return 0
        selector_args = (
            args
            if args.command
            in {
                "inspect-recording",
                "preprocess",
                "extract-features",
                "analyze-stability",
                "analyze-sensitivity",
                "qc",
                "run",
            }
            else None
        )
        selectors = _selectors(config, selector_args)
        if args.command == "list-recordings":
            records = discover_recordings(config.dataset.root, **selectors)
            if not records:
                print("No matching BIDS EEG recordings found.")
                return 0
            for record in records:
                print(record.label)
            print(f"Found {len(records)} recording(s).")
            return 0
        if args.command == "run":
            import mne

            with mne.use_log_level("ERROR"):
                execution = run_pipeline(
                    config,
                    **selectors,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
            summary = execution.summary()
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(f"Completed: {summary['recording']}")
                sampling = summary["sampling_frequency_hz"]
                channels = summary["eeg_channels"]
                epochs = summary["epochs"]
                print(
                    f"Sampling: {sampling['original']:g} -> {sampling['final']:g} Hz; "
                    f"eligible EEG channels: {channels['retained']}/{channels['original']}"
                )
                print(
                    f"Epochs: {epochs['retained']} retained, {epochs['rejected']} rejected "
                    f"of {epochs['initial']}"
                )
                print(
                    "Stages: "
                    + ", ".join(
                        f"{name}={details['status']}" for name, details in summary["stages"].items()
                    )
                )
                print("Feature outputs:")
                for name in (
                    "spectral_features",
                    "temporal_features",
                    "connectivity_features",
                ):
                    if name in summary["outputs"]:
                        print(f"- {name}: {summary['outputs'][name]}")
                print(f"Report: {summary['outputs']['report']}")
                print(f"Manifest: {summary['outputs']['execution_manifest']}")
                if summary["warnings"]:
                    print("Warnings:")
                    for warning in summary["warnings"]:
                        print(f"- {warning}")
            return 0
        recording = select_recording(config.dataset.root, **selectors)
        if args.command == "inspect-recording":
            print(json.dumps(inspect_recording(recording), indent=2, sort_keys=True))
            return 0
        source_raw = load_recording(recording)
        output_dir = (
            getattr(args, "output_dir", None)
            if getattr(args, "output_dir", None) is not None
            else config.dataset.derivatives_root / "quality-control"
        )
        overwrite = getattr(args, "overwrite", False) or config.reporting.overwrite
        if args.command == "analyze-sensitivity":
            sensitivity = analyze_preprocessing_sensitivity(source_raw, config, recording)
            paths = save_sensitivity_results(
                sensitivity, recording, output_dir, overwrite=overwrite
            )
            plot_paths: dict[str, str] = {}
            plot_warnings: list[str] = []
            if config.reporting.save_figures:
                figures = generate_diagnostic_figures(
                    source_raw,
                    sensitivity.baseline,
                    recording,
                    output_dir,
                    sensitivity=sensitivity,
                    dpi=config.reporting.figure_dpi,
                    overwrite=overwrite,
                )
                plot_paths = {name: str(path) for name, path in figures.paths.items()}
                plot_warnings = list(figures.warnings)
            summary = dict(sensitivity.comparison)
            summary["outputs"] = {name: str(path) for name, path in paths.items()}
            summary["figures"] = plot_paths
            summary["plot_warnings"] = plot_warnings
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        result = preprocess_raw(source_raw, config)
        if result.retained_epoch_count == 0:
            raise PreprocessingError("Preprocessing retained zero epochs.")
        if args.command == "extract-features":
            features = extract_features(result.epochs, config, recording)
            output_dir = (
                args.output_dir
                if args.output_dir is not None
                else config.dataset.derivatives_root / "features"
            )
            paths = save_feature_tables(
                features,
                recording,
                output_dir,
                overwrite=args.overwrite or config.reporting.overwrite,
            )
            summary = features.summary()
            summary["preprocessing"] = {
                "retained_epochs": result.retained_epoch_count,
                "retained_eeg_channels": result.retained_eeg_channel_count,
            }
            summary["outputs"] = {name: str(path) for name, path in paths.items()}
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command in {"analyze-stability", "qc"}:
            if args.command == "analyze-stability":
                without_connectivity = replace(
                    config, connectivity=replace(config.connectivity, enabled=False)
                )
                features = extract_features(result.epochs, without_connectivity, recording)
                stability = analyze_within_recording_stability(
                    features, config.reliability, config.epochs
                )
                paths = save_stability_results(
                    stability, recording, output_dir, overwrite=overwrite
                )
                plot_paths: dict[str, str] = {}
                plot_warnings: list[str] = []
                if config.reporting.save_figures:
                    figures = generate_diagnostic_figures(
                        source_raw,
                        result,
                        recording,
                        output_dir,
                        features=features,
                        stability=stability,
                        dpi=config.reporting.figure_dpi,
                        overwrite=overwrite,
                    )
                    plot_paths = {name: str(path) for name, path in figures.paths.items()}
                    plot_warnings = list(figures.warnings)
                summary = dict(stability.summary)
                summary["outputs"] = {name: str(path) for name, path in paths.items()}
                summary["figures"] = plot_paths
                summary["plot_warnings"] = plot_warnings
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 0 if stability.sufficient else 2
            if not config.reporting.qc_enabled:
                raise QCError("QC output is disabled in the configuration.")
            qc = collect_qc_metrics(result)
            metrics_path = save_qc_metrics(qc, recording, output_dir, overwrite=overwrite)
            plot_paths: dict[str, str] = {}
            plot_warnings: list[str] = []
            if config.reporting.save_figures:
                without_connectivity = replace(
                    config, connectivity=replace(config.connectivity, enabled=False)
                )
                features = extract_features(result.epochs, without_connectivity, recording)
                figures = generate_diagnostic_figures(
                    source_raw,
                    result,
                    recording,
                    output_dir,
                    features=features,
                    dpi=config.reporting.figure_dpi,
                    overwrite=overwrite,
                )
                plot_paths = {name: str(path) for name, path in figures.paths.items()}
                plot_warnings = list(figures.warnings)
            print(
                json.dumps(
                    {
                        "qc": qc.metrics,
                        "outputs": {"metrics": str(metrics_path)},
                        "figures": plot_paths,
                        "plot_warnings": plot_warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        summary = result.summary()
        if args.save:
            output_dir = (
                args.output_dir
                if args.output_dir is not None
                else config.dataset.derivatives_root / "preprocessing"
            )
            raw_path, epochs_path = save_preprocessing_result(
                result,
                recording,
                output_dir,
                overwrite=args.overwrite or config.reporting.overwrite,
            )
            summary["outputs"] = {
                "raw_fif": str(raw_path),
                "epochs_fif": str(epochs_path),
                "bids_derivative_compliant": False,
            }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (
        ConfigError,
        BIDSError,
        PreprocessingError,
        FeatureExtractionError,
        ConnectivityError,
        ReliabilityError,
        SensitivityError,
        QCError,
        PlotError,
        DerivativeError,
        ReportingError,
        PipelineExecutionError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
