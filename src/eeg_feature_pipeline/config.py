"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when pipeline configuration is missing or invalid."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a YAML mapping.")
    return value


def _only(values: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"Unknown setting(s) in '{name}': {', '.join(unknown)}.")


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive number.")


def _optional_entity(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(
            f"'{name}' must be a BIDS entity value, not a YAML boolean; quote values such as 'off'."
        )
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise ConfigError(f"'{name}' must be a non-empty BIDS entity value or null.")
    value = str(value).strip()
    for prefix in ("sub-", "ses-", "task-", "run-"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if not value or any(char in value for char in "/\\_"):
        raise ConfigError(f"'{name}' is not a valid BIDS entity value.")
    return value


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    derivatives_root: Path
    participant: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    highpass_hz: float = 1.0
    lowpass_hz: float = 45.0
    notch_hz: tuple[float, ...] = (50.0,)
    resample_hz: float | None = 250.0
    reference: str | tuple[str, ...] = "average"
    eeg_channels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EpochConfig:
    duration_seconds: float = 4.0
    overlap_seconds: float = 2.0
    reject_by_annotation: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    frequency_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    selected: tuple[str, ...] = (
        "absolute_band_power",
        "relative_band_power",
        "spectral_entropy",
    )
    temporal_selected: tuple[str, ...] = (
        "rms",
        "variance",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
    )
    analysis_range_hz: tuple[float, float] = (1.0, 45.0)
    psd_window_seconds: float = 2.0
    psd_overlap_fraction: float = 0.5
    psd_n_fft: int | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    reject_bad_channels: bool = True
    reject_bad_epochs: bool = True
    amplitude_threshold_uv: float = 150.0
    flat_threshold_uv: float = 0.5
    noisy_channel_zscore: float = 6.0
    ica_enabled: bool = False
    ica_n_components: int | float = 0.99
    ica_random_state: int = 97
    ica_max_iter: int = 500
    ica_fit_highpass_hz: float = 1.0
    ica_eog_detection: bool = True
    ica_eog_threshold: float = 3.0
    ica_min_eeg_channels: int = 4
    ica_min_duration_seconds: float = 20.0


@dataclass(frozen=True)
class ConnectivityConfig:
    enabled: bool = False
    methods: tuple[str, ...] = ("wpli2_debiased",)
    bands: tuple[str, ...] = ("alpha", "beta")
    minimum_epochs: int = 2
    include_diagonal: bool = False


@dataclass(frozen=True)
class ReliabilityConfig:
    enabled: bool = True
    segment_count: int = 2
    method: str = "spearman"
    minimum_epochs_per_group: int = 2


@dataclass(frozen=True)
class SensitivityConfig:
    enabled: bool = False
    filter_variants_hz: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ReportingConfig:
    output_formats: tuple[str, ...] = ("tsv", "json")
    save_figures: bool = True
    overwrite: bool = False
    figure_dpi: int = 150
    qc_enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    epochs: EpochConfig
    features: FeatureConfig
    artifacts: ArtifactConfig
    connectivity: ConnectivityConfig
    reliability: ReliabilityConfig
    sensitivity: SensitivityConfig
    reporting: ReportingConfig
    source_path: Path


_SECTIONS = {
    "dataset",
    "preprocessing",
    "epochs",
    "features",
    "artifacts",
    "connectivity",
    "reliability",
    "sensitivity",
    "reporting",
}


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"Required setting '{name}' must be a non-empty path.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _numbers(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, list):
        raise ConfigError(f"'{name}' must be a YAML list.")
    result: list[float] = []
    for value in values:
        _positive(value, name)
        result.append(float(value))
    return tuple(result)


def _strings(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise ConfigError(f"'{name}' must be a list of non-empty strings.")
    return tuple(values)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{name}' must be an integer of at least {minimum}.")
    return value


def _parse_bands(values: Any) -> dict[str, tuple[float, float]]:
    values = _mapping(values, "features.frequency_bands")
    bands: dict[str, tuple[float, float]] = {}
    for name, limits in values.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("Frequency-band names must be non-empty strings.")
        if name == "analysis_range":
            raise ConfigError("Frequency-band name 'analysis_range' is reserved.")
        if not isinstance(limits, list) or len(limits) != 2:
            raise ConfigError(f"Frequency band '{name}' must contain [low, high].")
        low, high = limits
        _positive(low, f"features.frequency_bands.{name}[0]")
        _positive(high, f"features.frequency_bands.{name}[1]")
        if low >= high:
            raise ConfigError(f"Frequency band '{name}' must have low < high.")
        bands[name] = (float(low), float(high))
    ordered = sorted((low, high, name) for name, (low, high) in bands.items())
    for (_, previous_high, previous_name), (low, _, name) in zip(
        ordered, ordered[1:], strict=False
    ):
        if low < previous_high:
            raise ConfigError(f"Frequency bands '{previous_name}' and '{name}' overlap.")
    if not bands:
        raise ConfigError("At least one frequency band is required.")
    return bands


_SPECTRAL_FEATURES = {
    "band_power",
    "absolute_band_power",
    "relative_band_power",
    "spectral_entropy",
}
_TEMPORAL_FEATURES = {
    "rms",
    "variance",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
}
_CONNECTIVITY_METHODS = {"coherence_squared", "pli", "wpli2_debiased"}


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in data:
        raise ConfigError(f"Missing required configuration section '{name}'.")
    return _mapping(data[name], name)


def load_config(path: str | Path, *, check_paths: bool = True) -> PipelineConfig:
    """Load and validate a pipeline YAML file.

    Relative paths are resolved relative to the configuration file. Validation is
    structural and conservative; signal-dependent checks remain the responsibility
    of later processing stages.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file: {exc}") from exc
    data = _mapping(loaded, "configuration")
    _only(data, _SECTIONS, "configuration")

    dataset_data = _section(data, "dataset")
    _only(
        dataset_data,
        {"root", "derivatives_root", "participant", "session", "task", "run"},
        "dataset",
    )
    root = _resolve_path(dataset_data.get("root"), base=source.parent, name="dataset.root")
    derivatives = _resolve_path(
        dataset_data.get("derivatives_root"), base=source.parent, name="dataset.derivatives_root"
    )
    if root == derivatives:
        raise ConfigError("'dataset.derivatives_root' must differ from the source dataset root.")
    if check_paths and not root.is_dir():
        raise ConfigError(f"BIDS dataset root does not exist or is not a directory: {root}")
    if derivatives.exists() and not derivatives.is_dir():
        raise ConfigError(f"Derivatives output path exists but is not a directory: {derivatives}")
    dataset = DatasetConfig(
        root=root,
        derivatives_root=derivatives,
        participant=_optional_entity(dataset_data.get("participant"), "dataset.participant"),
        session=_optional_entity(dataset_data.get("session"), "dataset.session"),
        task=_optional_entity(dataset_data.get("task"), "dataset.task"),
        run=_optional_entity(dataset_data.get("run"), "dataset.run"),
    )

    prep_data = _section(data, "preprocessing")
    _only(
        prep_data,
        {
            "highpass_hz",
            "lowpass_hz",
            "notch_hz",
            "resample_hz",
            "reference",
            "eeg_channels",
        },
        "preprocessing",
    )
    highpass = prep_data.get("highpass_hz", 1.0)
    lowpass = prep_data.get("lowpass_hz", 45.0)
    _positive(highpass, "preprocessing.highpass_hz")
    _positive(lowpass, "preprocessing.lowpass_hz")
    if highpass >= lowpass:
        raise ConfigError("'preprocessing.highpass_hz' must be below 'lowpass_hz'.")
    resample = prep_data.get("resample_hz", 250.0)
    if resample is not None:
        _positive(resample, "preprocessing.resample_hz")
        if resample <= 2 * lowpass:
            raise ConfigError(
                "'preprocessing.resample_hz' must exceed twice the low-pass frequency."
            )
    reference = prep_data.get("reference", "average")
    if isinstance(reference, list):
        reference = _strings(reference, "preprocessing.reference")
        if not reference:
            raise ConfigError("'preprocessing.reference' must not be an empty list.")
    elif not isinstance(reference, str) or not reference.strip():
        raise ConfigError(
            "'preprocessing.reference' must be 'average', 'none', a channel name, "
            "or a list of channel names."
        )
    else:
        reference = reference.strip()
    eeg_channels_value = prep_data.get("eeg_channels")
    eeg_channels = None
    if eeg_channels_value is not None:
        eeg_channels = _strings(eeg_channels_value, "preprocessing.eeg_channels")
        if not eeg_channels:
            raise ConfigError("'preprocessing.eeg_channels' must not be an empty list.")
        if len(set(eeg_channels)) != len(eeg_channels):
            raise ConfigError("'preprocessing.eeg_channels' must not contain duplicates.")
    preprocessing = PreprocessingConfig(
        highpass_hz=float(highpass),
        lowpass_hz=float(lowpass),
        notch_hz=_numbers(prep_data.get("notch_hz", [50.0]), "preprocessing.notch_hz"),
        resample_hz=None if resample is None else float(resample),
        reference=reference,
        eeg_channels=eeg_channels,
    )

    epoch_data = _section(data, "epochs")
    _only(
        epoch_data,
        {"duration_seconds", "overlap_seconds", "reject_by_annotation"},
        "epochs",
    )
    duration = epoch_data.get("duration_seconds", 4.0)
    overlap = epoch_data.get("overlap_seconds", 2.0)
    _positive(duration, "epochs.duration_seconds")
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or overlap < 0:
        raise ConfigError("'epochs.overlap_seconds' must be a non-negative number.")
    if overlap >= duration:
        raise ConfigError("'epochs.overlap_seconds' must be less than epoch duration.")
    reject_by_annotation = epoch_data.get("reject_by_annotation", True)
    if not isinstance(reject_by_annotation, bool):
        raise ConfigError("'epochs.reject_by_annotation' must be true or false.")
    epochs = EpochConfig(float(duration), float(overlap), reject_by_annotation)

    feature_data = _section(data, "features")
    _only(
        feature_data,
        {
            "frequency_bands",
            "selected",
            "temporal_selected",
            "analysis_range_hz",
            "psd_window_seconds",
            "psd_overlap_fraction",
            "psd_n_fft",
        },
        "features",
    )
    frequency_bands = _parse_bands(feature_data.get("frequency_bands"))
    selected = _strings(feature_data.get("selected", []), "features.selected")
    unsupported_spectral = sorted(set(selected) - _SPECTRAL_FEATURES)
    if unsupported_spectral:
        raise ConfigError("Unsupported spectral feature(s): " + ", ".join(unsupported_spectral))
    temporal_selected = _strings(
        feature_data.get(
            "temporal_selected",
            ["rms", "variance", "hjorth_activity", "hjorth_mobility", "hjorth_complexity"],
        ),
        "features.temporal_selected",
    )
    unsupported_temporal = sorted(set(temporal_selected) - _TEMPORAL_FEATURES)
    if unsupported_temporal:
        raise ConfigError("Unsupported temporal feature(s): " + ", ".join(unsupported_temporal))
    analysis_range = feature_data.get(
        "analysis_range_hz",
        [
            min(low for low, _ in frequency_bands.values()),
            max(high for _, high in frequency_bands.values()),
        ],
    )
    if not isinstance(analysis_range, list) or len(analysis_range) != 2:
        raise ConfigError("'features.analysis_range_hz' must contain [low, high].")
    analysis_low, analysis_high = analysis_range
    _positive(analysis_low, "features.analysis_range_hz[0]")
    _positive(analysis_high, "features.analysis_range_hz[1]")
    if analysis_low >= analysis_high:
        raise ConfigError("'features.analysis_range_hz' must have low < high.")
    if max(high for _, high in frequency_bands.values()) > preprocessing.lowpass_hz:
        raise ConfigError("Frequency bands cannot extend above 'preprocessing.lowpass_hz'.")
    if any(low < analysis_low or high > analysis_high for low, high in frequency_bands.values()):
        raise ConfigError("All feature bands must fall within 'features.analysis_range_hz'.")
    if analysis_high > preprocessing.lowpass_hz:
        raise ConfigError("Feature analysis range cannot extend above the preprocessing low-pass.")
    psd_window = feature_data.get("psd_window_seconds", 2.0)
    _positive(psd_window, "features.psd_window_seconds")
    if psd_window > epochs.duration_seconds:
        raise ConfigError("'features.psd_window_seconds' cannot exceed epoch duration.")
    psd_overlap = feature_data.get("psd_overlap_fraction", 0.5)
    if (
        isinstance(psd_overlap, bool)
        or not isinstance(psd_overlap, (int, float))
        or not 0 <= psd_overlap < 1
    ):
        raise ConfigError("'features.psd_overlap_fraction' must be in [0, 1).")
    psd_n_fft = feature_data.get("psd_n_fft")
    if psd_n_fft is not None:
        psd_n_fft = _integer(psd_n_fft, "features.psd_n_fft", minimum=2)
        if preprocessing.resample_hz is not None:
            window_samples = int(round(float(psd_window) * preprocessing.resample_hz))
            if psd_n_fft < window_samples:
                raise ConfigError(
                    "'features.psd_n_fft' cannot be shorter than the configured Welch window."
                )
    features = FeatureConfig(
        frequency_bands=frequency_bands,
        selected=selected,
        temporal_selected=temporal_selected,
        analysis_range_hz=(float(analysis_low), float(analysis_high)),
        psd_window_seconds=float(psd_window),
        psd_overlap_fraction=float(psd_overlap),
        psd_n_fft=psd_n_fft,
    )
    if not features.selected:
        raise ConfigError("'features.selected' must contain at least one spectral feature name.")
    if not features.temporal_selected:
        raise ConfigError("'features.temporal_selected' must contain at least one feature name.")
    artifact_data = _section(data, "artifacts")
    _only(
        artifact_data,
        {
            "reject_bad_channels",
            "reject_bad_epochs",
            "amplitude_threshold_uv",
            "flat_threshold_uv",
            "noisy_channel_zscore",
            "ica_enabled",
            "ica_n_components",
            "ica_random_state",
            "ica_max_iter",
            "ica_fit_highpass_hz",
            "ica_eog_detection",
            "ica_eog_threshold",
            "ica_min_eeg_channels",
            "ica_min_duration_seconds",
        },
        "artifacts",
    )
    threshold = artifact_data.get("amplitude_threshold_uv", 150.0)
    _positive(threshold, "artifacts.amplitude_threshold_uv")
    flat_threshold = artifact_data.get("flat_threshold_uv", 0.5)
    _positive(flat_threshold, "artifacts.flat_threshold_uv")
    if flat_threshold >= threshold:
        raise ConfigError("'artifacts.flat_threshold_uv' must be below 'amplitude_threshold_uv'.")
    noisy_zscore = artifact_data.get("noisy_channel_zscore", 6.0)
    _positive(noisy_zscore, "artifacts.noisy_channel_zscore")
    ica_n_components = artifact_data.get("ica_n_components", 0.99)
    if isinstance(ica_n_components, bool) or not isinstance(ica_n_components, (int, float)):
        raise ConfigError("'artifacts.ica_n_components' must be an integer or a fraction.")
    if isinstance(ica_n_components, int):
        if ica_n_components < 2:
            raise ConfigError("Integer 'artifacts.ica_n_components' must be at least 2.")
    elif not 0 < ica_n_components < 1:
        raise ConfigError("Fractional 'artifacts.ica_n_components' must be between 0 and 1.")
    ica_random_state = artifact_data.get("ica_random_state", 97)
    if isinstance(ica_random_state, bool) or not isinstance(ica_random_state, int):
        raise ConfigError("'artifacts.ica_random_state' must be an integer.")
    ica_max_iter = _integer(
        artifact_data.get("ica_max_iter", 500), "artifacts.ica_max_iter", minimum=1
    )
    ica_fit_highpass = artifact_data.get("ica_fit_highpass_hz", 1.0)
    _positive(ica_fit_highpass, "artifacts.ica_fit_highpass_hz")
    if ica_fit_highpass >= lowpass:
        raise ConfigError("'artifacts.ica_fit_highpass_hz' must be below the low-pass frequency.")
    ica_eog_threshold = artifact_data.get("ica_eog_threshold", 3.0)
    _positive(ica_eog_threshold, "artifacts.ica_eog_threshold")
    ica_min_eeg_channels = _integer(
        artifact_data.get("ica_min_eeg_channels", 4),
        "artifacts.ica_min_eeg_channels",
        minimum=2,
    )
    ica_min_duration = artifact_data.get("ica_min_duration_seconds", 20.0)
    _positive(ica_min_duration, "artifacts.ica_min_duration_seconds")
    artifacts = ArtifactConfig(
        reject_bad_channels=artifact_data.get("reject_bad_channels", True),
        reject_bad_epochs=artifact_data.get("reject_bad_epochs", True),
        amplitude_threshold_uv=float(threshold),
        flat_threshold_uv=float(flat_threshold),
        noisy_channel_zscore=float(noisy_zscore),
        ica_enabled=artifact_data.get("ica_enabled", False),
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_max_iter=ica_max_iter,
        ica_fit_highpass_hz=float(ica_fit_highpass),
        ica_eog_detection=artifact_data.get("ica_eog_detection", True),
        ica_eog_threshold=float(ica_eog_threshold),
        ica_min_eeg_channels=ica_min_eeg_channels,
        ica_min_duration_seconds=float(ica_min_duration),
    )

    connectivity_data = _section(data, "connectivity")
    _only(
        connectivity_data,
        {"enabled", "methods", "bands", "minimum_epochs", "include_diagonal"},
        "connectivity",
    )
    connectivity_methods = _strings(
        connectivity_data.get("methods", ["wpli2_debiased"]), "connectivity.methods"
    )
    unsupported_connectivity = sorted(set(connectivity_methods) - _CONNECTIVITY_METHODS)
    if unsupported_connectivity:
        raise ConfigError(
            "Unsupported connectivity method(s): " + ", ".join(unsupported_connectivity)
        )
    connectivity_bands = _strings(
        connectivity_data.get("bands", list(frequency_bands)), "connectivity.bands"
    )
    missing_bands = [name for name in connectivity_bands if name not in frequency_bands]
    if missing_bands:
        raise ConfigError(
            "Connectivity band(s) are not defined in features.frequency_bands: "
            + ", ".join(missing_bands)
        )
    minimum_epochs = _integer(
        connectivity_data.get("minimum_epochs", 2),
        "connectivity.minimum_epochs",
        minimum=2,
    )
    connectivity = ConnectivityConfig(
        enabled=connectivity_data.get("enabled", False),
        methods=connectivity_methods,
        bands=connectivity_bands,
        minimum_epochs=minimum_epochs,
        include_diagonal=connectivity_data.get("include_diagonal", False),
    )

    reliability_data = _section(data, "reliability")
    _only(
        reliability_data,
        {"enabled", "segment_count", "method", "minimum_epochs_per_group"},
        "reliability",
    )
    segment_count = reliability_data.get("segment_count", 2)
    if segment_count != 2 or isinstance(segment_count, bool):
        raise ConfigError(
            "'reliability.segment_count' must be 2 (an integer of at least 2; "
            "additional groups are unsupported)."
        )
    method = reliability_data.get("method", "spearman")
    if method not in {"none", "spearman"}:
        raise ConfigError("'reliability.method' must be none or spearman.")
    minimum_group_epochs = _integer(
        reliability_data.get("minimum_epochs_per_group", 2),
        "reliability.minimum_epochs_per_group",
        minimum=1,
    )
    reliability = ReliabilityConfig(
        reliability_data.get("enabled", True),
        segment_count,
        method,
        minimum_group_epochs,
    )

    sensitivity_data = _section(data, "sensitivity")
    _only(sensitivity_data, {"enabled", "filter_variants_hz"}, "sensitivity")
    variants_value = sensitivity_data.get("filter_variants_hz", [])
    if not isinstance(variants_value, list):
        raise ConfigError("'sensitivity.filter_variants_hz' must be a list of [low, high] pairs.")
    variants: list[tuple[float, float]] = []
    for index, pair in enumerate(variants_value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError(f"Sensitivity filter variant {index} must contain [low, high].")
        low, high = pair
        _positive(low, f"sensitivity.filter_variants_hz[{index}][0]")
        _positive(high, f"sensitivity.filter_variants_hz[{index}][1]")
        if low >= high:
            raise ConfigError(f"Sensitivity filter variant {index} must have low < high.")
        variants.append((float(low), float(high)))
    if len(variants) > 1:
        raise ConfigError(
            "'sensitivity.filter_variants_hz' supports one alternative passband only."
        )
    if sensitivity_data.get("enabled", False) and len(variants) != 1:
        raise ConfigError("Enabled sensitivity analysis requires exactly one filter variant.")
    if variants:
        alternative_low, alternative_high = variants[0]
        if alternative_low > features.analysis_range_hz[0]:
            raise ConfigError(
                "Sensitivity high-pass cannot exceed the feature analysis lower boundary."
            )
        if alternative_high < features.analysis_range_hz[1]:
            raise ConfigError(
                "Sensitivity low-pass cannot be below the feature analysis upper boundary."
            )
        if preprocessing.resample_hz is not None and (
            alternative_high >= preprocessing.resample_hz / 2.0
        ):
            raise ConfigError("Sensitivity low-pass must be below the resampled Nyquist frequency.")
    sensitivity = SensitivityConfig(sensitivity_data.get("enabled", False), tuple(variants))

    reporting_data = _section(data, "reporting")
    _only(
        reporting_data,
        {"output_formats", "save_figures", "figure_dpi", "qc_enabled", "overwrite"},
        "reporting",
    )
    figure_dpi = _integer(reporting_data.get("figure_dpi", 150), "reporting.figure_dpi", minimum=72)
    if figure_dpi > 600:
        raise ConfigError("'reporting.figure_dpi' must not exceed 600.")
    reporting = ReportingConfig(
        output_formats=_strings(
            reporting_data.get("output_formats", ["tsv", "json"]), "reporting.output_formats"
        ),
        save_figures=reporting_data.get("save_figures", True),
        figure_dpi=figure_dpi,
        qc_enabled=reporting_data.get("qc_enabled", True),
        overwrite=reporting_data.get("overwrite", False),
    )
    for section_name, values in {
        "artifacts": artifact_data,
        "connectivity": connectivity_data,
        "reliability": reliability_data,
        "sensitivity": sensitivity_data,
        "reporting": reporting_data,
    }.items():
        for key, value in values.items():
            if key in {
                "reject_bad_channels",
                "reject_bad_epochs",
                "ica_enabled",
                "ica_eog_detection",
                "enabled",
                "include_diagonal",
                "save_figures",
                "qc_enabled",
                "overwrite",
            } and not isinstance(value, bool):
                raise ConfigError(f"'{section_name}.{key}' must be true or false.")

    return PipelineConfig(
        dataset,
        preprocessing,
        epochs,
        features,
        artifacts,
        connectivity,
        reliability,
        sensitivity,
        reporting,
        source,
    )
