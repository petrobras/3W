---
name: add-demo-notebook
description: Add or update a Jupyter notebook in the 3W repository (toolkit demo, dataset overview, tutorial or benchmark) in the correct folder, with relative paths and execution requirements that keep bin/nb_test and the Sphinx docs working. Use when creating or editing .ipynb files.
---

# Add a 3W notebook

## 1. Choose the location

| Purpose | Location | Kept working across releases? |
|---------|----------|-------------------------------|
| How to use a toolkit feature | `toolkit/demos/NN_topic.ipynb` | Yes (BASIC) |
| Reference dataset overview | `dataset/demos/_basic/main.ipynb` | Yes (BASIC) |
| Contributor dataset overview / EDA | `dataset/demos/<author_name>/main.ipynb` | No |
| Tutorial or course material | `resources/<topic_or_version>/` | No |
| Complete, independent benchmark | `benchmarks/<name>/` (demo in `demo/_basic/`) | `_basic` only |

Never put notebooks in `docs/` (reserved for academic PDFs).

For `toolkit/demos/`, use the next free two-digit prefix and snake_case
(e.g. `15_windowing_examples.ipynb`). Every notebook there is copied into
the Sphinx tutorials automatically, so it must render cleanly.

## 2. Write it

- Start with a Markdown title and a short description of what the notebook
  shows; organize with numbered headings.
- Import from the package (`from ThreeWToolkit.dataset import
  ParquetDatasetConfig`), not from copied source code.
- Use paths relative to the notebook's folder, e.g.
  `dataset_path = "../../dataset"` from `toolkit/demos/` and
  `"../../../dataset"` from `dataset/demos/<author>/`. No absolute or
  user-specific paths.
- Keep it runnable top to bottom on CPU. Prefer small subsets
  (`target_class=[...]`, `event_type=[...]`, few epochs) so execution time
  stays reasonable, and state clearly when a cell is expensive.
- Fix random seeds where results matter.
- Write outputs to ignored folders (`output/`, `processed_data/`), never into
  `dataset/`.
- Do not add dependencies only for a notebook without also adding them to
  `pyproject.toml` (and running `uv lock`). Self-contained material in
  `resources/` may ship its own `requirements.txt` instead.

## 3. Clean before saving

- Restart the kernel and run all cells to confirm it works.
- Keep outputs only if they help readers (plots in demos); remove huge
  outputs, progress bars, warnings noise and local paths.
- Remove `.ipynb_checkpoints` and any secrets or tokens.
- Set a generic kernel (`python3`).

## 4. Verify

```bash
./bin/nb_test                           # BASIC notebooks (errors block release)
INCLUDE_NOT_BASIC=True ./bin/nb_test    # include other demos and resources
./bin/nb_test path/to/folder            # only notebooks under a folder
```

The dataset download (about 1.8 GB) happens on first run if `dataset/` is
incomplete. Ruff and mypy do not check notebooks, so review code cells
manually.

## 5. Link it

- New dataset overviews that should be showcased can be added to "Examples
  of Use" in the root `README.md`.
- Tutorials in `resources/` should have a `README.md` explaining the order
  of notebooks and how to run them.
