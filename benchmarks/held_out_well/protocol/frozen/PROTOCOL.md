# 3W-HW v1: strict held-out-well protocol for Petrobras 3W Dataset 2.0.0

Status: organizer proposal prepared by Sankalp Thakur (not an official 3W benchmark). Parameters in
section "Frozen parameters" were fixed in git commit A, before the split was drawn and before any model result existed.

## Why
The 3W repository removed its benchmark folds (`dataset/folds`, `problems/`) in PR #185 and the maintainers wrote in
issues #70 and #112 that new benchmarks "will need to include a suitable method for specifying which instances should be
included in the training and testing subsets" and "will need to include baselines". The old folds were per instance.
Instances of one well share instrumentation, units, operating regime and often overlap in time, so per-instance splits
leak. Carvalho (UFES 2021, Table 6, flow instability, 3W v1) measured the effect: random forest macro-F1 0.9928 (equal
folds) vs 0.6434 +- 0.2326 (folds by well).

## Data scope
- 3W Dataset 2.0.0 at petrobras/3W commit `d57c058af6baf65f1b285f3e56f46e4f018a07d9` (dataset tree `f41136c7ce407e33b759f5385c4f5cd7b3e376ce`), CC BY 4.0.
- Real instances only (`WELL-*`): 1,119 instances, 40 wells (WELL-00001 ... WELL-00042; 00017/00018 absent), 32,872,250 rows at 1 Hz, no timestamp gaps.
  Simulated (1,089) and hand-drawn (20) instances are not used in v1 for training or testing.
- The first 3,600 rows of every instance are unlabelled (NA). Labelled rows start at row 3,600.
- The folder name is not ground truth. 43 of 57 instances in folder 9 contain no event label at all; row labels decide.

## Labels
Row label y in {0..9, 101..109, NA}. Event class = y mod 100 (transient merged with its event). NA rows are never scored and never
reset online state. Episode = maximal run of one event class (adjacent transient + steady runs merged); episodes shorter than 300 rows
do not count as class support. `known_onset` = the run immediately before the episode is labelled normal inside the same instance.

## Decision grid (benchmark-defined, model independent)
Decisions at rows r_k = 3600 + 30k, k >= 0, r_k < n_rows. A model may use only rows r-3599 .. r of the same instance (causal, lookback cap 3,600 s).
Nothing from other instances of the test wells, no timestamps, file names, well ids, `class`, `state`, or future rows. Transductive use of
test-well data is not allowed except causal online adaptation inside the lookback cap.
Eligibility: a decision is eligible when >= 90 % of rows r-899 .. r have at least one finite value among the 18 process channels
(P-*, PT-P, QBS, QGL, T-*, ABER-*). Ineligible decisions are excluded from every denominator and reported (eligibility fractions).
Every eligible labelled decision of the evaluated wells must receive a finite prediction.

## Splits
`manifest.csv`: 40 wells -> 10 sealed test wells + 30 dev wells in 5 dev folds (6 wells each). Drawn by `src/w3/manifest.py` (hash ordering,
constraints K1-K5 in that file) from a public drand quicknet beacon round committed in git before the manifest existed.
- Model development, hyperparameter search and thresholds may use only dev wells (grouped 5-fold CV over dev folds).
- The sealed test wells are scored once per frozen model. Every test scoring is logged (`TEST-LOG.md`).
- Fixed-hyperparameter baselines are additionally reported as a 6-fold grouped CV (5 dev folds + test fold), which is unbiased only
  for models that were not tuned on any fold.

## Metrics (scorer: `src/w3/scorer.py`)
Window classification (10 classes, needs per-class prediction): confusion matrix on eligible labelled decisions; per-class F1; macro-F1 over classes
present in the evaluated truth; balanced accuracy; per-well macro-F1 mean and sd.
Event detection (needs one score per decision and one frozen threshold chosen on dev): alarm state machine with 2 consecutive decisions to switch on and
2 to switch off, reset on ineligible decisions. Reports: episode coverage (any alarm during the episode), carried-in share, onset detection (alarm rising edge within
300 s of a known onset), false alarm rising edges per eligible normal day, normal-time occupation, eligibility, unobservable episodes. Threshold-free: AUROC and AP of
the event score against (event class > 0).
Uncertainty: cluster bootstrap over wells (2,000 resamples, seed 611, percentile 95 % interval); paired bootstrap for model differences.

## Frozen parameters
stride 30 s; first decision row 3,600; lookback cap 3,600 rows; eligibility window 900 rows at 90 %; min episode 300 rows; onset tolerance 300 s;
hysteresis (2, 2); bootstrap B = 2,000, seed 611; test size 10 wells; dev folds 5.
