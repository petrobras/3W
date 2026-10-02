---
name: add-toolkit-component
description: Add a new component to the 3W Toolkit (ThreeWToolkit), such as a preprocessing step, feature extractor, model, trainer, metric or visualization, following the Pydantic config + build() pattern, with exports, tests and docs. Use when implementing or extending code under toolkit/ThreeWToolkit/.
---

# Add a 3W Toolkit component

## 1. Find the base class and a template

Pick the closest existing implementation and copy its structure instead of
inventing a new one.

| Component | Base classes (`ThreeWToolkit.core`) | Template |
|-----------|-------------------------------------|----------|
| Preprocessing step | `BasePreprocessing`, `BasePreprocessingConfig` | `preprocessing/normalize.py` |
| Feature extractor | `BaseFeatureExtractor`, `BaseFeatureExtractorConfig` | `feature_extraction/statistical.py` |
| Model | `BaseModels`, `ModelsConfig` | `models/sklearn_models.py`, `models/mlp.py` |
| Trainer | `BaseTrainer`, `BaseTrainerConfig` | `trainer/sklearn_trainer.py` |
| Assessment | `BaseAssessment`, `BaseAssessmentConfig` | `assessment/model_assess.py` |
| Dataset | `BaseDataset`, `BaseDatasetConfig` | `dataset/parquet_dataset.py` |

Read the base class in `toolkit/ThreeWToolkit/core/` first: it defines the
abstract methods you must implement.

## 2. Write the config and the component

Put both in the same file, config first:

```python
from pydantic import Field, PrivateAttr, field_validator

from ..core.base_dataset import BaseDataset
from ..core.base_preprocessing import BasePreprocessing, BasePreprocessingConfig
from ..core.dataset_outputs import DatasetOutputs


class ClipSignalsConfig(BasePreprocessingConfig):
    """Configuration for the ClipSignals preprocessing step."""

    lower: float = Field(default=0.0, description="Lower clipping bound.")
    upper: float = Field(default=1.0, description="Upper clipping bound.")
    _target: type = PrivateAttr(default_factory=lambda: ClipSignals)

    @field_validator("upper")
    @classmethod
    def validate_upper(cls, upper: float, info) -> float:
        if upper <= info.data["lower"]:
            raise ValueError("upper must be greater than lower.")
        return upper


class ClipSignals(BasePreprocessing):
    """Clip signal values to a fixed range.

    Attributes:
        config: Step configuration.
    """

    def __init__(self, config: ClipSignalsConfig):
        self.config: ClipSignalsConfig = config

    def fit(self, data: BaseDataset) -> None:
        """Nothing to fit for this step."""

    def transform(self, data: DatasetOutputs) -> DatasetOutputs:
        """Return a new DatasetOutputs with clipped signals."""
        signal = data.signal.clip(self.config.lower, self.config.upper)
        return DatasetOutputs(signal=signal, label=data.label, metadata=data.metadata)
```

Rules:

- `_target` is a `PrivateAttr` with a `lambda` so the class can be defined
  after the config. Users instantiate with `ClipSignalsConfig(...).build()`.
- Every parameter uses `Field(..., description=...)`; validation lives in
  `field_validator`s, not in `__init__`.
- Never mutate the input `DatasetOutputs` or its DataFrames: copy first or
  use operations that return new objects.
- Keep 3W categorical variables (valve states, see
  `_3W_CATEGORICAL_FEATURES` in `preprocessing/clean_signals.py`) out of
  numeric transforms unless the step is meant for them.
- Type hints everywhere, Google-style docstrings, mypy-clean for Python 3.10.

## 3. Export it

Add the config and class to the submodule `__init__.py` imports and
`__all__` (e.g. `toolkit/ThreeWToolkit/preprocessing/__init__.py`).

## 4. Test it

Create `tests/<submodule>/test_<module>.py`:

```python
import pytest

from ThreeWToolkit.preprocessing import ClipSignalsConfig


class TestClipSignalsConfig:
    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            ClipSignalsConfig(lower=1.0, upper=0.0)


class TestClipSignals:
    def test_clips_and_keeps_input(self, mock_dataset_factory):
        dataset = mock_dataset_factory(num_events=3, num_sensors=4, seed=0)
        step = ClipSignalsConfig(lower=-1.0, upper=1.0).build()
        event = dataset[0]
        original = event.signal.copy()

        result = step.transform(event)

        assert result.signal.max().max() <= 1.0
        assert event.signal.equals(original)
```

- Use `mock_dataset_factory` (in `tests/conftest.py`) for synthetic data.
  It accepts `num_events`, `num_sensors`, `num_timesteps_range`, NaN rates and
  `seed`.
- Never require files under `dataset/` in regular tests; CI has no Parquet
  files. Use `@pytest.mark.slow` if it is truly needed.
- Cover config validation, normal behavior, edge cases (NaNs, empty data)
  and input immutability.

## 5. Document it

- Docstrings are rendered by Sphinx. If the class belongs in the API
  reference, add it to the matching `autosummary` list in
  `toolkit/docs/source/api.rst`.
- Update the relevant page in `toolkit/docs/source/user_guide/` when the
  component is user-facing.
- Consider a short example in the matching `toolkit/demos/` notebook.

## 6. Verify

Run the `run-quality-checks` skill: at minimum the new test file,
`./bin/test` and `CI=false ./bin/lint`. If you added a dependency, update
`pyproject.toml` and run `uv lock`.
