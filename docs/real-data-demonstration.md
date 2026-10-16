# Real-data demonstration record

This document records the reproducible technical demonstration for
`eeg-feature-pipeline`. It is a single-recording execution example, not a clinical,
population-level, or comprehensive methodological validation.

## Source and selection

- Dataset: UC San Diego Resting State EEG Data from Patients with Parkinson's Disease
- OpenNeuro accession: `ds002778`
- Snapshot: `1.0.5`
- DOI: <https://doi.org/10.18112/openneuro.ds002778.v1.0.5>
- License: CC0
- Selected recording: `sub-pd6/ses-off/eeg/sub-pd6_ses-off_task-rest_eeg.bdf`
- Subject-specific subset: 19 files, 30,264,941 bytes on disk (30.3 MB reported by
  the downloader); raw BDF files account for the large majority.

The [official MNE-BIDS dataset-reading example](https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html)
documents the same subject-specific download and `ses-off/task-rest` selection. No
raw recording is included in this repository. The ignored `data/` directory
preserves the downloaded BIDS structure.

## Observed acquisition metadata

- Duration: 191.0 seconds (`97,792` samples)
- Sampling frequency: 512 Hz
- Acquisition bandwidth reported by MNE: 0-104 Hz
- Power-line frequency: 60 Hz
- Channels: 40 labeled EEG plus one stimulus channel
- Existing bad channels: none
- Annotations: two event annotations (`1` and `65536`), neither labeled `BAD`
- EOG channels: none declared
- Electrode-coordinate files: unavailable in the downloaded recording

The 40 EEG labels comprise 32 named scalp sensors and `EXG1` through `EXG8`. The
example configuration explicitly selects the 32 named scalp sensors. This leaves
the source BIDS channel metadata untouched.

## Effective processing choices

The complete configuration is [`../configs/example_real_eeg.yaml`](../configs/example_real_eeg.yaml).

- Passband: 1-45 Hz
- Line-noise handling: no notch; the 45 Hz low-pass excludes the reported 60 Hz line
- Reference: average across the selected eligible EEG channels
- Resampling: 512 to 250 Hz using MNE anti-alias filtering
- Automatic bad-channel screening: flat peak-to-peak <= 0.5 microvolts and robust
  first-difference noise z-score > 6
- ICA: disabled because no valid EOG channel is declared for the guarded automatic
  selection implemented by the pipeline
- Epochs: 4 seconds, 2-second overlap, no baseline correction
- Rejection: `BAD` annotations plus 200 microvolts peak-to-peak amplitude
- Welch: 2-second Hamming windows, 50% overlap, mean aggregation
- Bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
- Connectivity: recording-pooled magnitude-squared coherence and PLI, all five bands
- Stability: two chronological, temporally nonoverlapping groups; sensitivity disabled

The 200 microvolt criterion was selected after checking the signal distribution: the
median per-epoch maximum channel range was 194.4 microvolts. A 150 microvolt value
retained only 5 of 94 candidate windows, largely due to Fp1/Fp2; 200 microvolts
retained 57 while rejecting 37. This data-specific choice is documented rather than
interpreted as evidence of optimal preprocessing.

## Execution results

- Original / eligible EEG channels: 40 / 32
- Existing / automatically detected bad channels: 0 / 0
- ICA status: disabled; no candidate or removed components
- Candidate / retained / rejected epochs: 94 / 57 / 37
- Rejection fraction: 0.393617 (39.36%)
- Channels contributing rejection reasons: Fp1 (35), Fp2 (21), AF3 (4), F7 (2);
  counts can overlap within an epoch
- Processed channel RMS: median 10.020, minimum 5.088, maximum 35.540 microvolts
- Spectral table: 10,944 rows x 12 columns
- Temporal table: 1,824 rows x 11 columns
- Connectivity table: 4,960 rows x 14 columns
- Stability table: 512 rows x 14 columns
- Stability groups: 24 and 33 retained epochs; no boundary epoch required omission
- Pipeline warnings: one delta-band connectivity caution (four lower-edge cycles
  per epoch)
- Manifest status: `completed`
- Optional sensitivity stage: `skipped` by configuration

Feature rows retain participant, session, task, epoch/channel or channel-pair, band,
units, and analysis-level identifiers. Connectivity is explicitly labeled
`recording_pooled_epochs` with 57 contributing epochs.

## Descriptive numerical summary

| Band | Median absolute power (microvolts squared) | Median relative power | Median coherence squared | Median PLI |
|---|---:|---:|---:|---:|
| Delta | 24.8773 | 0.4409 | 0.5771 | 0.2659 |
| Theta | 7.0037 | 0.1223 | 0.2546 | 0.2466 |
| Alpha | 12.1649 | 0.2157 | 0.2978 | 0.3083 |
| Beta | 6.4226 | 0.1220 | 0.0802 | 0.1416 |
| Gamma | 1.2748 | 0.0215 | 0.0366 | 0.1309 |

Additional medians across retained epoch-channel rows were normalized spectral
entropy 0.7259, RMS 8.4283 microvolts, variance/Hjorth activity 70.9069 squared
microvolts, Hjorth mobility 0.2534, and Hjorth complexity 2.4139.

Within-recording feature-family median relative differences ranged from 1.94%
(spectral entropy) to 16.24% (absolute power). Guarded Spearman correlations across
matched channel/band vectors ranged from 0.869 to 0.974. Windows inside each group
can overlap, so these descriptive values are not independent-sample reliability
estimates.

## Sanity and integrity checks

- All evaluated absolute band powers were finite and nonnegative.
- Relative band powers were finite and in `[0, 1]`; the five bands cover the analysis
  range and summed from `0.9999999999999993` to `1.0000000000000002` per
  epoch-channel row (floating-point tolerance).
- Normalized spectral entropy values were finite and in `[0, 1]`.
- RMS, variance, and Hjorth values were finite where reported.
- All connectivity values were finite. Magnitude-squared coherence and PLI values
  were in `[0, 1]`, consistent with these estimators.
- Epoch and channel counts reconcile across preprocessing, feature tables, QC, and
  the execution manifest.
- The offline report is 384,213 bytes, embeds all five generated PNG figures as data
  URIs, and contains no project-local absolute path.
- All TSV result tables are nonempty and all JSON files parsed successfully.

Debiased squared wPLI was not configured for this example. The pipeline permits its
valid negative finite estimates and does not apply a generic nonnegative check to
that estimator.

## Generated figures and report

The completed run generated five readable figures: independent signal traces, a
compatible-range Welch PSD comparison, band-power summary, epoch retention, and
within-recording stability. Four unchanged PNGs are retained in `docs/assets/` for
the README. The local self-contained HTML report is reproducible from the command
below and remains ignored with the full derivatives.

```powershell
eeg-feature-pipeline run --config configs/example_real_eeg.yaml --subject pd6 --session off --task rest
```

The report and tables contain recording-level results only. They do not assert
clinical significance, diagnostic utility, anatomical connectivity, artifact-free
signals, or generalizability.
