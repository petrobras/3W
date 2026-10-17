# Held-out-well benchmark

## Scenario

Models are developed on some wells and deployed on wells they have never seen: a new well, or a well without labelled history. That is why this benchmark splits by well. Splitting by instance or by decision answers a different question, how well a model does on wells it has already seen, and it gives much higher numbers: with the same development data and the same LightGBM model, macro-F1 is 0.991 when decisions are split at random, 0.739 when instances are, and 0.295 when whole wells are held out. Instances of one well share instrumentation, units and operating regime, and often overlap in time, so an instance-level split leaks.

A benchmark for well-specific models (training and testing inside each well, split by time) is a natural companion. The decision grid and the scorer here could serve it, the split and the scenario would be different.

This folder is protocol 3W-HW v1. It follows [discussion #227](https://github.com/petrobras/3W/discussions/227).

## What is scored

At each decision a model gives an event score in [0, 1] (how likely any event is) and, if it classifies, a class from 0 to 9 (0 normal, 1 to 9 the event types of the 3W Dataset 2.0.0). A detector that gives only the score is scored on event detection alone.

**Primary metric: macro-F1** over the ten classes (normal and the nine event types), computed on the pooled eligible decisions of the 10 sealed test wells. Every class counts once, whatever its number of decisions. A class that does not occur in the pooled truth is not averaged; all ten occur in the sealed test wells. Reported with a 95 % interval from a bootstrap over wells.

Complementary metrics, all on the same pooled decisions unless stated:

| metric | definition |
|---|---|
| balanced accuracy | mean recall over the classes in the truth |
| confusion matrix, per-class F1 | rows are the truth, columns the prediction |
| macro-F1 over the nine event classes | as the primary metric, normal left out |
| macro-F1 per well | macro-F1 inside each test well (classes present in that well); mean and standard deviation over wells |
| event AUROC and average precision | the event score against "event class above 0", normal decisions against the rest, threshold free |
| episode coverage | share of the observable event episodes (at least 300 s long, with an eligible decision) during which the alarm is on at some decision |
| onset detected within 300 s | among episodes whose normal-to-event transition is observed inside the instance, share with an alarm rising edge within 300 s of the onset (an alarm that was already on does not count) |
| false alarms per normal day | alarm rising edges inside normal operation, per day of eligible normal time |
| normal time in alarm | share of eligible normal decisions with the alarm on |

The alarm is on after 2 consecutive eligible decisions with a score at or above a threshold and off after 2 consecutive eligible decisions below it. Ineligible decisions and the start of an instance reset it. The threshold is chosen on the development wells only, on out-of-fold scores: among the candidates with at most 1 false alarm per normal day and at most 5 % of normal time in alarm (on at least half a day of normal operation), the one with the highest episode coverage; a tie goes to the lower false alarm rate. Candidates are the quantiles of the normal scores from the median up, and a fixed grid from 0.05 to 0.99.

Intervals are percentile intervals from 2,000 resamples of whole wells (seed 611). `compare` reports paired differences on the same resamples.

## Protocol

Everything under [`protocol/`](protocol) is frozen for this version. Changing the grid, a metric definition, the class set or the wells defines a new protocol version.

* **Data.** The real instances of the 3W Dataset 2.0.0: 1,119 instances from 40 wells, 32,872,250 rows at 1 Hz. Simulated and hand-drawn instances are not used. Labels decide, not folder names (43 of the 57 instances in folder 9 contain no event label). A transient label (101 to 109) counts as its event class.
* **Decisions.** On a fixed grid: every 30 s of each instance from row 3,600, so that a full lookback exists (962,002 decisions). A model may use only the previous 3,600 rows of the same instance: no timestamps, file names, well identifiers, `class` or `state` columns, no future rows, nothing from other instances of a test well. Transductive use of test-well data is not allowed, except causal online adaptation inside the lookback.
* **Eligibility.** A decision is eligible when at least 90 % of the previous 900 rows have a finite value in at least one of the 18 process channels (961,578 decisions). Ineligible decisions are excluded from every denominator. Every eligible decision of the evaluated wells needs a finite prediction; the scorer rejects a file that lacks one.
* **Split.** 10 sealed test wells and 30 development wells in 5 folds of 6, drawn by a rule committed before the draw and seeded by a public [drand](https://drand.love) beacon round. [`COMMIT-A.md`](protocol/COMMIT-A.md) fixes the rule, its code and the label index by SHA-256 and names a round that lay in the future when it was written; [`COMMIT-B.md`](protocol/COMMIT-B.md) reveals the round's signature and the manifest (see [`protocol/`](protocol/README.md)). `python -m held_out_well manifest --check` regenerates [`manifest.csv`](protocol/manifest.csv) byte for byte; the test suite also runs the original files in [`protocol/frozen/`](protocol/frozen). Sealed test wells: 00001, 00004, 00011, 00013, 00015, 00023, 00026, 00035, 00037, 00041.
* **Development rules.** Models, hyper-parameters and thresholds use development wells only (grouped 5-fold cross-validation over the folds of the manifest). A frozen model is scored once on the sealed wells. For development, the macro averages run over the 8 classes that occur in at least two development wells (`dev_classes`).

Two properties of the test set matter when reading results. Five of the nine event classes (1, 3, 5, 6, 8) have a single test well, so per-class numbers on the sealed wells are anecdotes. And the pooled event metrics depend on the longest wells: well 00023 holds 45 % of the event decisions of the test set.

## Baselines

Self-evaluated by the author of this folder, who also drew the split and wrote the scorer: **organizer baselines**. All 40 real wells were used in the author's earlier private experiments, so these entries are not independent results, and a new split seed would not remove that exposure for the author's own models.

Sealed test wells, 95 % well-bootstrap intervals in brackets. The numbers come from `python -m held_out_well score` on the saved predictions of each entry. The predictions (24 MB) are not part of the repository; `results/scores.json` holds the scores in full, with the SHA-256 of each prediction file.

| entry | code | macro-F1 (primary) | balanced accuracy | macro-F1, 9 event classes | event AUROC | event AP | episode coverage | onset within 300 s | false alarms / normal day |
|---|---|---|---|---|---|---|---|---|---|
| `simple` (z-test, no training) | this folder | - | - | - | 0.472 [0.369, 0.765] | 0.447 [0.217, 0.758] | 20.9 % [4.6, 89.7] | 50.0 % [5.6, 78.4] | 1.06 [0.25, 2.96] |
| `volatility` (one-class) | this folder | - | - | - | 0.537 [0.328, 0.833] | 0.563 [0.237, 0.817] | 19.1 % [2.9, 86.7] | 16.7 % [0.0, 27.8] | 0.41 [0.05, 1.39] |
| `rf` (random forest) | this folder | 0.266 [0.145, 0.499] | 0.307 [0.220, 0.539] | 0.213 [0.060, 0.454] | 0.740 [0.536, 0.933] | 0.733 [0.632, 0.833] | 30.9 % [11.6, 86.7] | 33.3 % [0.0, 56.0] | 2.94 [0.04, 13.77] |
| `lgbm` (LightGBM) | this folder | 0.292 [0.202, 0.493] | 0.277 [0.195, 0.506] | 0.242 [0.116, 0.447] | 0.581 [0.308, 0.968] | 0.626 [0.356, 0.918] | 60.0 % [12.5, 85.0] | 40.0 % [0.0, 67.6] | 0.39 [0.00, 2.20] |
| `small_lgbm` (dev-selected small LightGBM) | earlier repository | 0.258 [0.164, 0.483] | 0.349 [0.246, 0.546] | 0.208 [0.072, 0.439] | 0.751 [0.504, 0.959] | 0.726 [0.570, 0.905] | 65.5 % [29.4, 87.8] | 40.0 % [0.0, 67.6] | 0.55 [0.00, 3.13] |
| `event_detector` (dev-selected binary detector) | earlier repository | - | - | - | 0.449 [0.236, 0.913] | 0.423 [0.249, 0.796] | 5.5 % [1.6, 32.3] | 0.0 % [0.0, 0.0] | 0.18 [0.00, 1.16] |
| `tspulse` (TSPulse, zero-shot) | earlier repository | - | - | - | 0.493 [0.346, 0.565] | 0.440 [0.127, 0.751] | 14.5 % [4.0, 56.4] | 26.7 % [0.0, 45.3] | 0.96 [0.23, 1.51] |

How the entries were made:

* The four reference baselines are reproduced by the code in [`held_out_well/baselines`](held_out_well/baselines), with fixed hyper-parameters and nothing tuned on any fold. `simple` is a windowed z-test of the last 5 minutes against the previous 55 minutes of the same instance (no training). `volatility` scores the 15-minute variability of each process channel against its 99th percentile in normal operation (fitted on normal decisions only). `rf` is a random forest (100 trees) and `lgbm` a LightGBM multiclass model (200 rounds), both on 333 causal window features. Class probabilities give the event score as 1 minus P(normal).
* `small_lgbm` and `event_detector` come from a development study of a small LightGBM (200 trees, 3,000 leaves, 28 KB as an int8 export) and of a binary event detector (called `best` and `best_binary` in the frozen configuration). Their configuration was frozen on the development wells before the sealed wells were scored; see [`results/configs`](results/configs) (SHA-256 of `dev_selected_config.json`: `131ce334...`). `tspulse` is the pretrained time-series model [TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1) (IBM Granite, Apache-2.0) used zero-shot as a reconstruction-error detector, with the same threshold rule. The code of these three entries is in the author's earlier repository, [3w-held-out-well](https://github.com/sankalpsthakur/3w-held-out-well/tree/dd8cddc), and is not part of this folder; their predictions are scored here by the same scorer.
* With ten test wells the intervals are wide and the entries cannot be told apart: every paired difference between `small_lgbm` and the two supervised baselines has an interval that contains 0 for macro-F1.

Reading the table: `-` means the entry gives no class, so the classification metrics are not defined for it. Event AUROC and average precision are computed on scores quantised to 4,096 logit bins; the unquantised values (`auroc_exact` and `average_precision_exact` in `results/scores.json`) differ by at most 0.0001 (AUROC) and 0.0007 (average precision) for these entries. The alarm threshold of each entry is in `results/scores.json`. The threshold rule, not only the score, decides the alarm metrics: the same scores give other coverage and false-alarm numbers under another rule.

## Run it

From the repository root, with the 3W Toolkit installed (`uv pip install -e '.[dev]'`, see the [contributing guide](../../3W_TOOLKIT_CONTRIBUTING.md)):

```bash
cd benchmarks/held_out_well
python -m pytest -q                              # unit tests, no dataset needed
python -m held_out_well manifest --check         # regenerate the split, compare byte for byte
python -m held_out_well score preds.npz          # score a prediction file on the sealed test wells
python -m held_out_well score preds.npz --split dev
python -m held_out_well compare a.npz b.npz      # paired differences with intervals
```

A prediction file is an `.npz` with `score` (one value in [0, 1] per decision, in the order of `protocol/decisions.npz`) and, for a classifier, `pred` (integer class 0 to 9). Development predictions must be out-of-fold: the file holds the predictions of the 30 development wells made by models that did not train on them, and those of the 10 test wells made by a model trained on the development wells. The alarm threshold is chosen from the development part of the same file, or given with `--theta`.

With a local copy of the dataset (the folder with the sub-folders `0` to `9`; the 3W Toolkit downloads it when it is missing):

```bash
uv pip install -r requirements.txt                                   # LightGBM, for the baselines only
python -m held_out_well rebuild-index --dataset-root ../../dataset --out out/rebuild
python -m held_out_well baselines --dataset-root ../../dataset --out out/baselines
python -m held_out_well score out/baselines/preds_lgbm.npz
```

`rebuild-index` reads every real instance with the toolkit loader and compares the label runs, the eligibility and the decision table with the committed files (instances, label runs, hashes of the decision arrays). The baselines run takes about 15 minutes on 4 CPU cores in a Kaggle kernel (features 1 minute, random forest 3, LightGBM 9, bootstrap intervals included).

## What comes from the 3W Toolkit

| part | source |
|---|---|
| reading instances, labels, file selection | `ThreeWToolkit.dataset.ParquetDataset` |
| macro-F1, balanced accuracy, accuracy, per-class F1, AUROC, average precision | `ThreeWToolkit.metrics`, called with sample weights (see below) |
| well split from a committed beacon draw | this folder: the toolkit splitters split instances, not wells |
| decision grid and eligibility | this folder |
| alarm, episode, onset and false-alarm metrics | this folder: the toolkit has no time-related metrics yet (`BACKLOG.md` lists them as wanted) |
| bootstrap over wells | this folder |

The toolkit metrics accept `sample_weight`, so a metric of the pooled decisions of any set of wells, and of every bootstrap resample, is one call on a summed confusion matrix or on summed score histograms. The tests check these calls against closed-form formulas and, on raw decisions, against resampling the decisions themselves. Event scores are quantised to 4,096 logit bins in the histograms; `auroc_exact` and `average_precision_exact` in the reports are the same toolkit functions on the unquantised scores.

## Files

| path | content |
|---|---|
| [`held_out_well/`](held_out_well) | the package: `manifest` (split), `grid` (decision grid), `benchmark` (committed tables), `detection` (alarms, episodes), `scoring` (toolkit metrics, bootstrap, threshold), `loader` (toolkit loader, index rebuild), `cli`, `baselines` |
| [`protocol/`](protocol) | frozen artefacts: manifest, drand round, commit records, label index, decision table; `frozen/` holds the files that were committed before the draw |
| [`results/`](results) | scores of the entries above, frozen configurations, SHA-256 of the saved predictions |
| [`tests/`](tests) | unit tests (the repository's CI runs only the top-level `tests/`; run these as shown above) |

## Licences and attribution

Code in this folder: Apache-2.0, like the repository. The files in `protocol/` that are derived from the labels of the 3W Dataset 2.0.0 (`index_labels.json`, `instances.csv`, `decisions.npz`, `manifest.csv` and the files that describe them) carry its CC BY 4.0 licence. Please cite the dataset:

> Vaz Vargas, R.E., de Melo Junior, A.J., de Campos Lima, C.B. et al. 3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells. Sci Data 13, 949. https://doi.org/10.1038/s41597-026-07225-z (2026).

and the project, see [`CITATION.md`](../../CITATION.md).
