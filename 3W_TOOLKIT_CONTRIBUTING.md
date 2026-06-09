# 🤝 Contributing Guide

Welcome to our **3W Toolkit**! 🎉 We're excited that you're interested in contributing. This guide will help you understand our modular architecture and how to extend the toolkit with new models, preprocessing steps, feature extractors, trainers, and assessments.

We welcome contributions in all forms and truly appreciate every effort. 💙 Helping the community goes beyond writing code — answering questions, supporting other users, and improving documentation are just as important.

You can also support the project by spreading the word. 📣 Share it in blog posts about projects built with the library, talk about it on social media when it helps you, or simply leave a ⭐️ on the repository to show your support.

No matter how you choose to contribute, please be respectful and follow our [code of conduct](#code-of-conduct). 📜

## Summary of the Guidelines

- **One pull request per feature or bug fix** - Keep PRs focused and atomic
- **Follow the architecture patterns** - Use base classes and existing implementations as templates
- **Use Pydantic configs** - All new components should support configuration-driven instantiation
- **Include tests and documentation** - Essential for all contributions
- **Run quality checks before submitting** - Use `./bin/lint` and ensure `./bin/test` passes
- **Write clear commit messages** - Follow conventional commits format


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Architecture Overview](#architecture-overview)
- [How to Contribute](#how-to-contribute)
  - [Adding a New Model](#adding-a-new-model)
  - [Creating Preprocessing Steps](#creating-preprocessing-steps)
  - [Creating Feature Extractors](#creating-feature-extractors)
  - [Creating Trainers](#creating-trainers)
  - [Creating Assessment Strategies](#creating-assessment-strategies)
- [Development Workflow](#development-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style Guidelines](#code-style-guidelines)

---

## Code of Conduct

This project adheres to a [code of conduct](https://github.com/petrobras/3W/blob/main/CODE_OF_CONDUCT.md) that we expect all contributors to follow. Please be respectful, inclusive, and considerate in all interactions.

## Getting Started

### Prerequisites
- Python >= 3.10
- Git and GitHub knowledge
- Familiarity with Pydantic for configuration

### Setting Up Your Development Environment

1. **Clone and navigate to repository:**
   ```bash
   git clone https://github.com/petrobras/3W.git
   cd 3W
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e '.[dev]'
   # or with uv:
   uv sync --all-extras
   ```

4. **Verify installation:**
   ```bash
   ./bin/test  # Run tests
   ./bin/lint  # Run linting
   ```

---

## Architecture Overview

The 3W Toolkit follows a **modular, plugin-based architecture** using base classes and the **Strategy Pattern** to enable extensibility.

### 1. Core Principles

**Separation of Concerns:**
- **Data Loading & Management** (`BaseDataset`): Handles dataset operations
- **Preprocessing** (`BasePreprocessing`): Data transformation steps (non-destructive)
- **Feature Extraction** (`BaseFeatureExtractor`): Feature engineering
- **Models** (`BaseModels`): Model architecture and inference
- **Training** (`BaseTrainer`): Training logic and orchestration
- **Assessment** (`BaseAssessment`): Evaluation and visualization

**Configuration-Driven:**
- All components use **Pydantic configs** for type-safe configuration
- Configs support dynamic instantiation via `_target` attribute (Instantiable pattern)
- Field validators ensure configuration correctness

**Lazy Evaluation:**
- `BaseDataset` and transformations support property-based lazy evaluation
- Prevents unnecessary data loading and copying

### 2. Base Class Hierarchy

```
ThreeWToolkit/
├── core/
│   ├── base_models.py                    # BaseModels (abstract model interface)
│   ├── base_trainer.py                   # BaseTrainer (training orchestration)
│   ├── base_dataset.py                   # BaseDataset (data management)
│   ├── base_preprocessing.py             # BasePreprocessing (transform steps)
│   ├── base_feature_extractor.py         # BaseFeatureExtractor (feature engineering)
│   ├── base_assessment.py                # BaseAssessment (evaluation)
│   ├── base_pipeline.py                  # BasePipeline (pipeline orchestration)
│   ├── base_transform.py                 # BaseTransform (transform interface)
│   ├── base_instantiable.py              # Instantiable (config-driven instantiation)
│   └── enums.py                          # Shared enums
├── models/
│   ├── mlp.py                            # PyTorch MLP
│   ├── sklearn_models.py                 # Scikit-learn wrappers
│   └── torch_models.py                   # Additional PyTorch models
├── trainer/
│   ├── sklearn_trainer.py                # Scikit-learn trainer
│   └── torch_trainer.py                  # PyTorch trainer
├── preprocessing/
│   ├── clean_signals.py
│   ├── impute_missing.py
│   ├── normalize.py
│   ├── remap.py
│   ├── fill_labels.py
│   └── ... (modular steps)
├── feature_extraction/
│   ├── statistical.py
│   ├── wavelet.py
│   ├── exponential_statistics.py
│   ├── windowing.py
│   └── adapters.py
├── assessment/
│   └── model_assess.py
└── dataset/
    ├── parquet_dataset.py
    ├── subset_dataset.py
    └── transform_dataset.py
```

### 3. Key Patterns

**BaseModels:**
```python
class BaseModels(ABC):
    """All models inherit from this."""
    
    @abstractmethod
    def save(self, filename: str | Path) -> Path:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filename: str | Path) -> "BaseModels":
        """Load model from disk."""
        pass
    
    def predict(self, X):
        """Make predictions."""
        pass
```

**BaseTrainer:**
```python
class BaseTrainer(ABC):
    """Framework-agnostic training orchestration."""
    
    @abstractmethod
    def fit(self, dataset, **kwargs) -> TrainingResult:
        """Train the model."""
        pass
```

**BasePreprocessing / BaseFeatureExtractor:**
```python
class BasePreprocessing(ABC, BaseTransform):
    """Non-destructive data transformation."""
    
    @abstractmethod
    def fit_and_transform(
        self, dataset: BaseDataset
    ) -> DatasetOutputs:
        """Fit and transform, returning new DatasetOutputs."""
        pass
```

**Configuration with Pydantic:**
```python
from pydantic import BaseModel, Field, field_validator
from ThreeWToolkit.core import Instantiable

class MyComponentConfig(BaseModel, Instantiable):
    """Configuration for custom component."""
    
    _target: type["MyComponent"]  # For dynamic instantiation
    param1: int = Field(..., description="First parameter", gt=0)
    param2: float = Field(default=0.5, description="Second parameter")
    
    @field_validator("param1")
    @classmethod
    def validate_param1(cls, v):
        if v < 1:
            raise ValueError("param1 must be >= 1")
        return v
```

---

## How to Contribute

There are several ways you can contribute to this toolkit:

### 🐛 Fix Outstanding Issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#submitting-pull-requests) and open a Pull Request!

We have labeled some issues as **Good First Issue** - these are beginner-friendly and a great way to start contributing to open-source.

### 🆕 Add New Models

New ML models are constantly being released. If you want to implement a new model:
- Check if someone is already working on it (search existing issues/PRs)
- Open an issue describing the model and link to the paper
- Follow our [Adding a New Model](#adding-a-new-model) guide

### 🔧 Create Preprocessing Steps

Implement new data preprocessing operations:
- Data cleaning, normalization, imputation, remapping
- Non-destructive transformations (create new data, don't modify original)
- Follow our [Creating Preprocessing Steps](#creating-preprocessing-steps) guide

### 📊 Create Feature Extractors

Implement new feature engineering methods:
- Statistical features, wavelets, domain-specific features
- Support windowing and overlapping operations
- Follow our [Creating Feature Extractors](#creating-feature-extractors) guide

### 🚂 Create Trainers

Implement framework-specific training logic:
- Support for new ML frameworks (XGBoost, LightGBM, etc.)
- Cross-validation integration
- Follow our [Creating Trainers](#creating-trainers) guide

### 📈 Create Assessment Strategies

Implement new evaluation and visualization methods:
- Custom metrics and evaluation functions
- Specialized visualization strategies
- Follow our [Creating Assessment Strategies](#creating-assessment-strategies) guide

### 📚 Improve Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add usage examples
- Improve docstrings
- Translate documentation

### 🐞 Report Bugs

Before reporting a bug:
1. **Search existing issues** to avoid duplicates
2. **Verify it's not your code** - Ask in Discussions if unsure
3. **Provide minimal reproducible example** (< 30 lines if possible)

Include in your bug report:
- OS and Python version
- Library version
- Full traceback
- Minimal code to reproduce

### 💡 Request Features

When requesting a new feature:
1. Describe the **motivation** - What problem does it solve?
2. Provide **detailed description** of the proposed feature
3. Include a **code snippet** showing desired usage
4. Link to **relevant papers** if applicable

> All contributions are equally valuable to the community. 🥰

---

### Adding a New Model

To add a new model (e.g., XGBoost, LightGBM, custom neural networks), follow these steps:

#### Step 1: Choose Model Category

Determine which trainer your model needs:
- **Scikit-learn compatible**: Use `SklearnTrainer` (models with `.fit()` and `.predict()`)
- **PyTorch**: Use `TorchTrainer` (models extending `torch.nn.Module`)
- **Custom**: Create custom trainer if needed

#### Step 2: Create a Configuration Class

Create a configuration class in your model file:

```python
from pydantic import BaseModel, Field, field_validator
from ThreeWToolkit.core import ModelsConfig, Instantiable

class YourModelConfig(ModelsConfig, Instantiable):
    """Configuration for YourModel."""
    
    _target: type["YourModel"]
    
    param1: int = Field(..., description="First hyperparameter", gt=0)
    param2: float = Field(default=0.1, description="Second hyperparameter")
    learning_rate: float = Field(default=0.01, description="Learning rate", gt=0)
    
    @field_validator("param1")
    @classmethod
    def validate_param1(cls, v):
        if v < 1:
            raise ValueError("param1 must be >= 1")
        return v
```

#### Step 3: Implement the Model Class

Inherit from `BaseModels` and implement required methods:

**For Scikit-learn models:**
```python
from ThreeWToolkit.core import BaseModels
import pickle

class YourModel(BaseModels):
    """Scikit-learn style model."""
    
    def __init__(self, config: YourModelConfig):
        self.config = config
        self._model = None  # Initialize actual sklearn model
        # self._model = SomeSklearnEstimator(param1=config.param1, ...)
    
    @property
    def model_name(self) -> str:
        return "YourModel"
    
    def save(self, filename: str | Path) -> Path:
        """Save model using pickle."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._model, f)
        return path
    
    @classmethod
    def load(cls, filename: str | Path) -> "YourModel":
        """Load model from pickle."""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        instance = cls(YourModelConfig(param1=1))
        instance._model = model
        return instance
    
    def predict(self, X):
        """Make predictions."""
        return self._model.predict(X)
```

**For PyTorch models:**
```python
import torch
from torch import nn
from ThreeWToolkit.core import BaseModels

class YourTorchModel(BaseModels, nn.Module):
    """PyTorch model."""
    
    def __init__(self, config: YourModelConfig):
        super().__init__()
        nn.Module.__init__(self)
        
        self.config = config
        # Define layers
        self.fc1 = nn.Linear(10, config.param1)
        self.fc2 = nn.Linear(config.param1, 2)
    
    @property
    def model_name(self) -> str:
        return "YourTorchModel"
    
    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, filename: str | Path) -> Path:
        """Save model state dict."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path
    
    @classmethod
    def load(cls, filename: str | Path) -> "YourTorchModel":
        """Load model from state dict."""
        instance = cls(YourModelConfig(param1=10))
        instance.load_state_dict(torch.load(filename))
        return instance
    
    def predict(self, X):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            output = self.forward(X)
        return output.cpu().numpy()
```

#### Step 4: Register in Models Module

Add to `toolkit/ThreeWToolkit/models/__init__.py`:

```python
from .your_model import YourModel, YourModelConfig

__all__ = [
    "YourModel",
    "YourModelConfig",
    # ... existing exports
]
```

#### Step 5: Add Tests

Create `tests/models/test_your_model.py`:

```python
import pytest
import numpy as np
from ThreeWToolkit.models import YourModel, YourModelConfig

class TestYourModel:
    @pytest.fixture
    def config(self):
        return YourModelConfig(param1=10, param2=0.1)
    
    @pytest.fixture
    def model(self, config):
        return YourModel(config)
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "YourModel"
        assert model.config.param1 == 10
    
    def test_predict(self, model):
        """Test prediction."""
        X = np.random.randn(5, 10)
        predictions = model.predict(X)
        assert predictions.shape[0] == 5
    
    def test_save_load(self, model, tmp_path):
        """Test save and load."""
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = YourModel.load(path)
        assert loaded.model_name == model.model_name
```

---

### Creating Preprocessing Steps

Preprocessing steps perform non-destructive data transformations. Create a new step by inheriting from `BasePreprocessing`:

#### Step 1: Create Configuration

```python
from pydantic import BaseModel, Field
from ThreeWToolkit.core import BasePreprocessingConfig, Instantiable

class YourPreprocessingConfig(BasePreprocessingConfig, Instantiable):
    """Configuration for YourPreprocessing."""
    
    _target: type["YourPreprocessing"]
    param1: float = Field(default=0.1, description="Threshold")
    column_names: list[str] | None = Field(
        default=None, description="Columns to process"
    )
```

#### Step 2: Implement the Class

```python
from ThreeWToolkit.core import BasePreprocessing, DatasetOutputs

class YourPreprocessing(BasePreprocessing):
    """Custom preprocessing step."""
    
    def __init__(self, config: YourPreprocessingConfig):
        super().__init__(config)
        self.config = config
        self._fitted_stats = None
    
    def fit_and_transform(
        self, dataset: BaseDataset
    ) -> DatasetOutputs:
        """
        Fit and transform data (non-destructive).
        
        Returns:
            DatasetOutputs with transformed data
        """
        signal_df = dataset.signal_df
        label_df = dataset.label_df
        
        # Compute statistics on original data (fit)
        self._fitted_stats = signal_df.describe()
        
        # Create transformed copy (non-destructive)
        transformed_signal = signal_df.copy()
        # Apply transformation...
        
        return DatasetOutputs(
            signal=transformed_signal,
            label=label_df,
            metadata=dataset.metadata
        )
```

#### Step 3: Add Tests

```python
def test_your_preprocessing(dataset):
    """Test preprocessing step."""
    config = YourPreprocessingConfig(param1=0.1)
    preprocessing = YourPreprocessing(config)
    
    result = preprocessing.fit_and_transform(dataset)
    
    # Original dataset unchanged
    assert dataset.signal_df is not result.signal
    
    # Result has correct shape
    assert result.signal.shape == dataset.signal_df.shape
```

---

### Creating Feature Extractors

Feature extractors inherit from `BaseFeatureExtractor`. They should support windowing and overlap operations:

```python
from ThreeWToolkit.core import (
    BaseFeatureExtractor,
    WindowSizeMixin,
    OverlapOffsetMixin,
    Instantiable
)
from pydantic import Field

class YourFeatureExtractorConfig(BaseModel, Instantiable):
    """Configuration for feature extractor."""
    
    _target: type["YourFeatureExtractor"]
    window_size: int = Field(default=100, description="Window size")
    overlap_offset: int = Field(default=50, description="Overlap offset")

class YourFeatureExtractor(
    BaseFeatureExtractor, 
    WindowSizeMixin, 
    OverlapOffsetMixin
):
    """Extract custom features from windowed data."""
    
    def __init__(self, config: YourFeatureExtractorConfig):
        super().__init__(config)
        self.window_size = config.window_size
        self.overlap_offset = config.overlap_offset
    
    def extract_features(self, signal_df) -> np.ndarray:
        """Extract features from signal."""
        # Apply windowing and feature extraction
        pass
```

---

### Creating Trainers

Create framework-specific trainers by inheriting from `BaseTrainer`:

```python
from ThreeWToolkit.core import BaseTrainer, TrainingResult, TrainingHistory

class YourFrameworkTrainer(BaseTrainer):
    """Trainer for YourFramework models."""
    
    def fit(
        self,
        model: BaseModels,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> TrainingResult:
        """
        Train the model.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training parameters
        
        Returns:
            TrainingResult with trained model and history
        """
        # Training logic here
        history = TrainingHistory(
            train_loss=[...],
            val_loss=[...]
        )
        
        return TrainingResult(
            model=model,
            history=history,
            metadata={}
        )
```

---

### Creating Assessment Strategies

Assessment strategies evaluate model performance:

```python
from ThreeWToolkit.core import BaseAssessment, AssessmentOutput

class YourAssessment(BaseAssessment):
    """Custom assessment strategy."""
    
    def assess(
        self,
        y_true,
        y_pred,
        **kwargs
    ) -> AssessmentOutput:
        """
        Assess model predictions.
        
        Returns:
            AssessmentOutput with metrics and visualizations
        """
        metrics = {
            "accuracy": np.mean(y_true == y_pred),
            # ... more metrics
        }
        
        return AssessmentOutput(
            metrics=metrics,
            visualizations={}
        )
```

---

## Development Workflow

### Branch Naming Convention

- `feature/model-name` - For new models
- `feature/preprocessing-name` - For new preprocessing steps
- `feature/feature-extractor-name` - For new feature extractors
- `bugfix/issue-description` - For bug fixes
- `docs/topic` - For documentation updates
- `refactor/component-name` - For refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed.
List key changes:
- Change 1
- Change 2

Fixes #issue_number
```

**Types:**
- `feat` - New feature or component
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style changes (formatting, imports)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks
- `perf` - Performance improvements

**Examples:**
```
feat(models): add XGBoost model implementation

- Create XGBoostConfig with hyperparameter validation
- Implement XGBoostModel wrapper for sklearn interface
- Add sklearn_trainer support
- Include comprehensive tests for training and prediction

Closes #42
```

```
fix(preprocessing): prevent in-place modifications in normalize

- Clone data before normalization
- Add epsilon handling for zero divisions
- Update tests to verify non-destructive behavior

Fixes #135
```

---

## Testing Guidelines

### Required Tests

Every contribution should include tests:

1. **Unit Tests** for your component:
   - Configuration validation
   - Initialization with various parameters
   - Core functionality
   - Error handling for edge cases

2. **Integration Tests**:
   - Full pipeline execution
   - Cross-validation (if supported)
   - Model training and prediction together

3. **Data Integrity Tests** (important for preprocessing/feature extraction):
   - Original data is not modified (for transformations)
   - Output shapes and types are correct
   - Handles missing/invalid data gracefully

### Running Tests

```bash
# Run all tests with coverage
./bin/test

# Run specific test file
pytest tests/models/test_your_model.py

# Run single test
pytest tests/models/test_your_model.py::TestYourModel::test_predict

# Run with verbose output
pytest tests/models/test_your_model.py -v

# Run with coverage report
pytest tests/models/test_your_model.py --cov=toolkit/ThreeWToolkit/models
```

### Test Structure

Place tests alongside the module they test:
```
toolkit/ThreeWToolkit/models/your_model.py
tests/models/test_your_model.py
```

Use class-based organization for related tests:
```python
class TestYourModel:
    @pytest.fixture
    def config(self):
        return YourModelConfig(...)
    
    @pytest.fixture
    def model(self, config):
        return YourModel(config)
    
    def test_initialization(self, model):
        ...
    
    def test_predict(self, model):
        ...

class TestYourModelConfig:
    def test_validation(self):
        ...
```

### Fixtures

Common fixtures are defined in `tests/conftest.py`. Create local fixtures in test files as needed:

```python
@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    signal = np.random.randn(100, 5)
    label = np.random.randint(0, 2, 100)
    return BaseDataset(signal=signal, label=label)
```

---

## Submitting Pull Requests

### Before Submitting

- [ ] Code follows style guidelines (`./bin/lint` passes)
- [ ] All tests pass (`./bin/test`)
- [ ] New tests added for new functionality
- [ ] Documentation updated and docstrings complete
- [ ] No merge conflicts with main/dev branch
- [ ] Commit messages follow convention
- [ ] Related issues are linked

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] New model
- [ ] New preprocessing step
- [ ] New feature extractor
- [ ] New trainer
- [ ] Bug fix
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Component Details
- **Component Name**: [Name of model/extractor/etc.]
- **Base Class**: [BaseModels/BasePreprocessing/etc.]
- **Framework**: [PyTorch/Scikit-learn/Other]
- **Supported Tasks**: [Classification/Regression/Both]

## Testing
- [ ] Unit tests added and passing
- [ ] Integration tests added and passing
- [ ] Edge cases handled
- [ ] Data integrity verified (for transformations)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Type hints are complete
- [ ] Documentation/docstrings updated
- [ ] No breaking changes (or justified breaking changes)
- [ ] Related issues linked
- [ ] `./bin/lint` passes
- [ ] `./bin/test` passes

## Notes for Reviewers
Any additional context or notes for reviewers.
```

### Review Process

1. **Automated Checks**: Tests, linting, type checking run automatically
2. **Code Review**: Maintainers review code for:
   - Architecture alignment
   - Code quality and style
   - Test coverage
   - Documentation completeness
3. **Address Comments**: Respond to review comments
4. **Approval & Merge**: Maintainers approve and merge

---

## Code Style Guidelines

### Python Style

- Follow **PEP 8**
- Use **type hints** for all function parameters and returns
- Maximum line length: **88 characters** (Black formatter)
- Use **meaningful variable names**
- Prefer **explicit over implicit**

### Imports

Organize imports in three groups (separated by blank lines):
```python
# 1. Standard library
import json
from pathlib import Path
from typing import Any

# 2. Third-party libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# 3. Local imports
from ThreeWToolkit.core import BaseModels, BaseModelsConfig
from ThreeWToolkit.utils import general_utils
```

### Type Hints

Always include type hints:

```python
# ❌ Bad
def fit_and_transform(self, dataset):
    return dataset.transform()

# ✅ Good
def fit_and_transform(self, dataset: BaseDataset) -> DatasetOutputs:
    """Fit and transform dataset."""
    return DatasetOutputs(signal=transformed, label=dataset.label_df)
```

Use `TypeAlias` for complex types:

```python
from typing import TypeAlias

SignalType: TypeAlias = np.ndarray | pd.DataFrame
```

### Docstrings

Use **Google-style docstrings** (3 empty lines after class definition):

```python
class YourClass:
    """Brief one-liner description.
    
    Longer description if needed, explaining the purpose and usage.
    
    Attributes:
        param1: Description of param1.
        param2: Description of param2.
    """
    
    def method(self, x: int, y: str) -> bool:
        """Brief description of method.
        
        Longer description explaining what the method does.
        
        Args:
            x: Description of x parameter.
            y: Description of y parameter.
        
        Returns:
            Description of return value.
        
        Raises:
            ValueError: When x is negative.
            TypeError: When y is not a string.
        
        Example:
            >>> obj = YourClass()
            >>> result = obj.method(5, "hello")
            >>> print(result)
            True
        """
        pass
```

### Linting and Formatting

Before committing, run:

```bash
# Format code with Black and fix with Ruff
./bin/lint

# Or manually:
black --extend-exclude '\.ipynb$' toolkit tests
ruff check --fix toolkit tests
mypy toolkit tests
```

**Key tools:**
- **Black** - Code formatter (line length: 88)
- **Ruff** - Fast linter and auto-fixer
- **Mypy** - Static type checker

### Configuration Examples

```python
# ✅ Good Pydantic config
from pydantic import BaseModel, Field, field_validator

class MyConfig(BaseModel):
    """Configuration for MyComponent."""
    
    learning_rate: float = Field(
        default=0.01,
        description="Learning rate for optimization",
        gt=0,
        le=1
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for training",
        gt=0,
        le=512
    )
    num_epochs: int = Field(
        default=100,
        description="Number of training epochs",
        ge=1
    )
    
    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v):
        if v > 0.1:
            raise ValueError("learning_rate should typically be <= 0.1")
        return v
```

---

## Getting Help

### Resources

- **Documentation**: Check `/docs` folder and README files
- **Examples**: See `/toolkit/demos/` for usage examples
- **Tests**: Review existing tests for patterns and best practices
- **Core Classes**: Study `toolkit/ThreeWToolkit/core/` for interfaces

### Contact

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Pull Requests**: For code contributions

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0 for code, CC BY 4.0 for data).

---

Thank you for contributing! 🎉