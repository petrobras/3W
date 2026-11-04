---
name: load-3w-dataset
description: Load, filter and inspect 3W Dataset instances (Parquet files of offshore oil well time series) with ThreeWToolkit's ParquetDataset, including event classes, sources (real, simulated, drawn), variables and transient labels. Use when writing code, analyses or notebooks that read the 3W Dataset.
---

# Load the 3W Dataset

## Dataset facts

- Location: `dataset/<label>/<source>_<id>.parquet`, 2228 files in
  version 2.0.0. Metadata lives in `dataset/dataset.ini`.
- Folder = event label:

  | Label | Event | Transient |
  |-------|-------|-----------|
  | 0 | Normal operation | – |
  | 1 | Abrupt increase of BSW | yes |
  | 2 | Spurious closure of DHSV | yes |
  | 3 | Severe slugging | no |
  | 4 | Flow instability | no |
  | 5 | Rapid productivity loss | yes |
  | 6 | Quick restriction in PCK | yes |
  | 7 | Scaling in PCK | yes |
  | 8 | Hydrate in production line | yes |
  | 9 | Hydrate in service line | yes |

- Source from the filename prefix: `WELL-` (real), `SIMULATED_`, `DRAWN_`
  (hand-drawn).
- Each file: `timestamp` index (1 s sampling), float sensor columns, plus
  `class` (per-observation label) and `state` (operational status), both
  pandas nullable integers that may contain `<NA>`. In `class`, `0` marks
  normal periods, `label + 100` the transient period before the event is
  established (`TRANSIENT_OFFSET = 100`), and `label` the steady event.
- Variables (units in `dataset.ini`): pressures in Pa (`P-PDG`, `P-TPT`,
  `P-MON-CKP`, ...), temperatures in °C (`T-TPT`, `T-JUS-CKP`, ...), flow
  rates in m³/s (`QGL`, `QBS`), choke openings in % (`ABER-CKP`,
  `ABER-CKGL`) and valve states in {0, 0.5, 1} (`ESTADO-*`, categorical).
  Not every instance has every variable, and many columns can be all-NaN.

## Loading with the toolkit

```python
from ThreeWToolkit.dataset import ParquetDatasetConfig

ds = ParquetDatasetConfig(
    path="dataset",                 # repo dataset folder or a download path
    target_class=[0, 4],            # list of labels; None loads all
    event_type=["real"],            # "real", "simulated", "drawn"; None = all
    columns=["P-PDG", "P-TPT", "T-TPT"],  # None loads every signal column
    target_column="class",          # None keeps `class` inside `signal`
).build()

print(len(ds))                      # number of instances
event = ds[0]                       # DatasetOutputs
event.signal                        # pandas DataFrame of variables
event.label                         # pandas Series from `class`
event.metadata                      # file_name, event_class, event_type
```

- `path` must contain the full dataset. If the expected number of Parquet
  files is not found, the loader **downloads it from Figshare** (about
  1.8 GB) into `path`. Do not trigger this in unit tests.
- `force_download=True` deletes and re-downloads `path`. Never point it at
  the repository's `dataset/` folder.
- `split` only supports `None` or `"list"` (with `file_list` of paths
  relative to `path`, e.g. `"4/WELL-00001_20170316110203.parquet"`); train,
  val and test splits are not implemented. Use `utils.data_splitter` for
  splitting.
- Iterating (`for event in ds`) loads files lazily, one at a time. Avoid
  materializing all instances in memory unless needed.
- For clustering, `ds.load_instances_by_variable(["P-PDG"])` returns
  `{variable: [np.ndarray, ...]}`.

## Common next steps

- Clean and normalize with `ThreeWToolkit.preprocessing` (`CleanSignals`,
  `ImputeMissing`, `Normalize`, `FillLabels`, `RemapClass`).
- Apply a transform to a whole dataset lazily with
  `TransformedDataset(ds, step.transform)` from `ThreeWToolkit.dataset`.
- Plot with `ThreeWToolkit.data_visualization` (e.g. the 3W chart).
- See `toolkit/demos/01_dataset_examples.ipynb` and
  `dataset/demos/_basic/main.ipynb` for complete examples.

## Rules

- Treat files under `dataset/` as read-only (CC BY 4.0 data, versioned
  separately). Write derived data to ignored folders such as `output/` or
  `processed_data/`.
- In tests, use `mock_dataset_factory` from `tests/conftest.py` instead of
  real files, or mark the test `@pytest.mark.slow`.
