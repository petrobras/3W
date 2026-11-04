---
name: run-quality-checks
description: Run and fix the 3W repository quality checks (pytest via bin/test, mypy + ruff via bin/lint, notebook execution via bin/nb_test) the same way CI does. Use before finishing any code change, or when tests, lint, type checks or notebooks fail.
---

# Run 3W quality checks

All commands run from the repository root inside the uv environment
(`uv sync --locked --extra dev`, then `source .venv/bin/activate` or prefix
commands with `uv run`).

## 1. Tests: `./bin/test`

```bash
pytest tests/<submodule>/test_<module>.py -v   # fast loop on what you changed
./bin/test                                     # full suite + coverage, as CI
RUN_SLOW_TESTS=1 ./bin/test                    # also tests marked `slow`
```

- `bin/test` adds `-m "not slow"` unless `RUN_SLOW_TESTS=1`. Tests marked
  `slow` read the real Parquet dataset.
- `pytest.ini` sets `pythonpath = toolkit`, so tests import `ThreeWToolkit`
  from the working tree.
- CI runs on Python 3.10, 3.11 and 3.12 with a checkout that **excludes
  `*.parquet`**. A test that passes locally but touches `dataset/` will fail
  in CI: replace it with `mock_dataset_factory` or mark it `slow`.
- It writes `coverage.xml`; do not commit it.

## 2. Lint and types: `./bin/lint`

```bash
CI=false ./bin/lint    # local: mypy, ruff check --fix, lychee (if installed)
./bin/lint             # CI mode (default): mypy, ruff check without fixes
```

- mypy targets Python 3.10 (`mypy_path = toolkit/`). Use `X | None`, avoid
  syntax newer than 3.10, and prefer precise types over `Any` or
  `# type: ignore`.
- ruff line length is 88. `ruff format toolkit tests` is optional but keeps
  diffs consistent with the codebase.
- Notebooks (`*.ipynb`) are excluded from mypy and ruff.
- lychee checks links in Markdown files; it is skipped when not installed
  and always in CI mode.

## 3. Notebooks: `./bin/nb_test` (manual, slow)

Not run by CI. Use it when you change code that demos rely on or edit a
notebook.

```bash
./bin/nb_test                          # BASIC: toolkit/demos, dataset/demos/_basic
INCLUDE_NOT_BASIC=True ./bin/nb_test   # also other dataset demos and resources/
./bin/nb_test toolkit/demos            # any path, reported as warnings
```

- BASIC failures exit non-zero and block releases; NOT_BASIC failures are
  warnings only.
- Many notebooks download the dataset (about 1.8 GB) or train models and can
  take a long time. The path argument must be a directory; to run a single
  notebook directly, use
  `python -m nbconvert --to notebook --execute --output-dir /tmp <file>`
  (nbconvert runs the kernel in the notebook's folder, so relative paths
  resolve as in `bin/nb_test`).
- Execution logs are kept in a temporary directory printed at the start of
  the run; read the failing notebook's log for the traceback.

## 4. Dependency changes

If `pyproject.toml` dependencies changed, run `uv lock` and include the
updated `uv.lock`. CI installs with `uv sync --locked` and fails when the
lock file is stale.

## Reporting

Report which checks you ran and their results. If something fails for
reasons unrelated to your change (e.g. network access for downloads), say
so explicitly instead of skipping silently.
