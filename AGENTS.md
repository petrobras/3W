# AGENTS.md

Instructions for AI coding agents working in the 3W repository. This is the
single source of truth shared by every agent. Claude Code, GitHub Copilot,
Cursor and Codex read this file directly; `GEMINI.md` only points here.
Human contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) and
[3W_TOOLKIT_CONTRIBUTING.md](3W_TOOLKIT_CONTRIBUTING.md); this file
summarizes them and does not replace them.

## Project overview

The 3W Project, maintained by Petrobras, supports Machine Learning research on
detecting and classifying undesirable events in offshore oil wells. It has two
independently versioned parts:

- **3W Dataset** (`dataset/`, version in `dataset/dataset.ini`): multivariate
  time series stored as Parquet files, one instance per file.
- **3W Toolkit** (`toolkit/ThreeWToolkit/`, version in `pyproject.toml`): the
  `ThreeWToolkit` Python package, published on PyPI.

## Repository layout

| Path | Content |
|------|---------|
| `toolkit/ThreeWToolkit/` | Importable package ([submodules](3W_TOOLKIT_STRUCTURE.md)) |
| `toolkit/demos/` | Numbered toolkit demo notebooks (also rendered in the docs) |
| `toolkit/docs/` | Sphinx documentation (Read the Docs) |
| `tests/` | pytest suite, mirroring the toolkit submodules |
| `dataset/0` … `dataset/9` | Parquet instances; folder name is the event label |
| `dataset/dataset.ini` | Variable descriptions, event names, labels, settings |
| `dataset/demos/` | Dataset overviews (`_basic/` is the reference one) |
| `resources/` | Self-contained tutorials and course material |
| `docs/` | Academic PDFs only (theses, dissertations). No notebooks |
| `bin/` | `test`, `lint`, `nb_test`, `sync_agent_files` helper scripts |
| `.agents/skills/` | Agent skills, edited only here (see "Agent skills") |

The toolkit submodules and what each one holds are described in
[3W_TOOLKIT_STRUCTURE.md](3W_TOOLKIT_STRUCTURE.md). Alongside them,
`pipeline.py` orchestrates end-to-end training and cross-validation.

## Setup and commands

Python >= 3.10. Dependencies are managed with [uv](https://docs.astral.sh/uv/)
and pinned in `uv.lock`.

```bash
uv sync --locked --extra dev        # dev environment (.venv)
uv sync --locked --all-extras       # every extra in pyproject.toml
./bin/test                          # pytest + coverage, skips `slow` tests
RUN_SLOW_TESTS=1 ./bin/test         # include tests that read the real dataset
pytest tests/preprocessing/test_normalize.py -k l2   # targeted run
CI=false ./bin/lint                 # mypy + ruff check --fix (local mode)
./bin/nb_test                       # execute BASIC demo notebooks (slow)
```

`./bin/lint` defaults to CI mode (no auto-fix). `./bin/test` and `./bin/lint`
are the checks CI runs on Python 3.10, 3.11 and 3.12; both must pass.

## Architecture conventions

- **Config + component pairs.** Every component has a Pydantic config
  subclassing the matching base config from `ThreeWToolkit.core`
  (`BasePreprocessingConfig`, `BaseFeatureExtractorConfig`, `ModelsConfig`,
  `BaseDatasetConfig`, ...). The config declares its component with
  `_target: type = PrivateAttr(default_factory=lambda: MyComponent)` and is
  instantiated with `MyComponentConfig(...).build()`.
- **Data contract.** Datasets yield `DatasetOutputs` (`signal: DataFrame`,
  `label: Series | None`, `metadata: dict`). Preprocessing steps and feature
  extractors implement `transform(DatasetOutputs) -> DatasetOutputs`
  (plus optional `fit(BaseDataset)`) and must **not** mutate their input:
  copy before modifying.
- **Exports.** Register new public classes in the submodule `__init__.py`
  and its `__all__`.
- **Validation** belongs in `field_validator`s on the config, with `Field(...,
  description=...)` for every parameter.

## Code style

- Ruff (line length 88), mypy with `python_version = 3.10`: use `X | None`,
  built-in generics, and type hints on all public signatures.
- Google-style docstrings in English; docstring/comment lines up to 72 chars.
- Import groups: standard library, third-party, local (relative imports
  inside the package).
- Prefer meaningful variable names; do not introduce `Any` where a concrete
  type is known.

## Testing rules

- Put tests in `tests/<submodule>/test_<module>.py`, class-based
  (`class TestX:`), using fixtures from `tests/conftest.py`, especially
  `mock_dataset_factory` for synthetic `DatasetOutputs` datasets.
- Unit tests must not depend on the real dataset: CI checks out the repo
  **without** Parquet files. Mark anything that needs them with
  `@pytest.mark.slow`.
- Cover config validation, core behavior, edge cases (NaNs, empty inputs)
  and, for transforms, that the input is left unchanged.
- Matplotlib runs with the `Agg` backend in tests; do not call `plt.show()`.

## Dataset rules

The file layout and the Parquet storage conventions (index, dtypes,
compression) are described in
[3W_DATASET_STRUCTURE.md](3W_DATASET_STRUCTURE.md). On top of those:

- Never modify, rename, move or regenerate files under `dataset/0`–`9`; they
  are licensed CC BY 4.0 and versioned separately from the code.
- Filename prefix gives the source: `WELL-` (real), `SIMULATED_`, `DRAWN_`.
- Labels: `0` is normal operation; `1`–`9` are event types (names in
  `dataset.ini`). Transient periods use `label + 100` (`TRANSIENT_OFFSET`).
- Load data through `ParquetDatasetConfig(path=..., target_class=[...],
  event_type=[...]).build()` rather than ad-hoc `pd.read_parquet` loops in
  toolkit code. The loader downloads the dataset from Figshare when missing.

## Notebooks

- Toolkit usage demos: `toolkit/demos/NN_topic.ipynb`.
- Dataset overviews: `dataset/demos/<author>/main.ipynb`.
- Tutorials: `resources/<topic>/`. Benchmarks: `benchmarks/<name>/`.
- Notebooks in `toolkit/demos/` and `dataset/demos/_basic/` must keep running
  across releases (`./bin/nb_test`). Ruff and mypy skip `*.ipynb`.

## Dependencies, versioning and contributions

- After changing dependencies in `pyproject.toml`, run `uv lock` and keep
  `uv.lock` in sync.
- Versions follow SemVer and are bumped manually by maintainers; see
  [VERSIONING.md](VERSIONING.md). Do not bump versions unless asked.
- Pull requests target the **`dev`** branch, never `main`. One focused change
  per PR, conventional commit messages (`feat(preprocessing): ...`).
- Do not add generated artifacts (`output/`, `coverage.xml`, executed
  notebook copies, downloaded dataset zips) to the repository.

## Agent skills

Reusable task guides live in `.agents/skills/<name>/SKILL.md`:

- `add-toolkit-component`: add a preprocessing step, feature extractor,
  model, metric or other toolkit component with config, tests and docs.
- `run-quality-checks`: run and fix lint, tests and notebook checks.
- `load-3w-dataset`: load, filter and inspect 3W Dataset instances.
- `add-demo-notebook`: add a demo or overview notebook in the right place.

`.agents/skills/` is the only place to edit skills. `.claude/skills/` is a
generated copy for agents that do not read `.agents/skills/`: never edit it
directly. After changing a skill, run `./bin/sync_agent_files`
(CI runs it with `--check`).
