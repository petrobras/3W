# 🤝 Contributing Guide

Welcome to our **3W Toolkit**! 🎉 We're excited that you're interested in contributing. This guide will help you understand our architecture and how to add new models, training strategies, and prediction strategies.

We welcome contributions in all forms and truly appreciate every effort. 💙 Helping the community goes beyond writing code — answering questions, supporting other users, and improving documentation are just as important.

You can also support the project by spreading the word. 📣 Share it in blog posts about projects built with the library, talk about it on social media when it helps you, or simply leave a ⭐️ on the repository to show your support.

No matter how you choose to contribute, please be respectful and follow our [code of conduct](#code-of-conduct). 📜

## Summary of the Guidelines

- **One pull request per feature or bug fix** - Keep PRs focused and atomic
- **Follow the architecture patterns** - Use existing models as templates
- **Include tests and documentation** - Essential for all contributions
- **Run quality checks before submitting** - Use `bin/lint` and ensure tests pass
- **Write clear commit messages** - Follow conventional commits format


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Architecture Overview](#architecture-overview)
- [How to Contribute](#how-to-contribute)
  - [Adding a New Model](#adding-a-new-model)
  - [Creating Training Strategies](#creating-training-strategies)
  - [Creating Prediction Strategies](#creating-prediction-strategies)
- [Development Workflow](#development-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style Guidelines](#code-style-guidelines)

---

## Code of Conduct

This project adheres to a [code of conduct](https://github.com/petrobras/3W/blob/main/CODE_OF_CONDUCT.md) that we expect all contributors to follow. Please be respectful, inclusive, and considerate in all interactions.

## Getting Started
### Prerequisites
### Setting Up Your Development Environment

---

## Architecture Overview

Our toolkit is built on some core principles:

### 1. Separation of Concerns

The architecture separates:
-
-
-
- **Model Definition** (architecture and forward pass)
- **Training Logic** (how the model learns)
- **Prediction Logic** (how the model makes predictions)

### 2. Strategy Pattern

We use the Strategy Pattern to make components pluggable:

```
BaseModels (Abstract)
├── Defines model architecture
├── get_training_strategy() → Returns compatible training strategy
└── get_prediction_strategy() → Returns compatible prediction strategy

TrainingStrategy (Abstract)
└── Implements training loop

PredictionStrategy (Abstract)
└── Implements prediction logic
```

### 3. Configuration-Driven

All models are configured using **Pydantic** for:
- Type safety
- Automatic validation
- Clear documentation
- IDE support

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

### ⚡ Create New Strategies

Implement new training or prediction strategies:
- Novel training loops (e.g., curriculum learning, adversarial training)
- Specialized inference methods (e.g., beam search, ensemble predictions)
- Follow our [Creating Training Strategies](#creating-training-strategies) guide

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

To add a new model (e.g., XGBoost, LightGBM, custom neural networks), follow these steps carefully:

#### Step 1: Register the Model Type

Add your model to the `ModelTypeEnum` in `core/enums.py`:

```python
class ModelTypeEnum(Enum):
    # ... existing models
    XGBOOST = "xgboost"  # Your new model
```

**Naming conventions:**
- Use lowercase with underscores (e.g., `xgboost`, `light_gbm`)
- Keep names concise but descriptive
- Follow existing patterns in the enum

#### Step 2: Create a Configuration Class

Create a new configuration class that:
- Inherits from `ModelsConfig`
- Defines all hyperparameters with Pydantic `Field`
- Includes validation via `@field_validator`
- Has clear descriptions for each parameter
- Sets `target_` to reference your model class

**Key Points:**
- Use descriptive field names
- Set sensible defaults
- Add constraints (e.g., `gt=0` for positive values)
- Include docstrings with examples

#### Step 3: Implement the Model Class

Your model class must:
- Inherit from `BaseModels` (and `nn.Module` if using PyTorch)
- Implement all abstract methods:
  - `forward(x)` - Model forward pass (can be empty for sklearn-like models)
  - `save(path)` - Serialize model to disk
  - `load(path)` - Load model from disk
  - `get_training_strategy()` - Return compatible training strategy class
  - `get_prediction_strategy()` - Return compatible prediction strategy class
- Override `model_name` property for logging

**Important Considerations:**
- For PyTorch models: Call both `super().__init__(config)` and `nn.Module.__init__()`
- For sklearn-like models: Wrap the estimator in your model class
- Handle device placement for PyTorch models
- Consider dynamic input size inference when applicable

#### Step 4: Update Registry

Add your model to the allowed models list in `core/base_models.py` validator.

```python
@field_validator("model_type")
@classmethod
def check_model_type(cls, v, info):
    allowed = {
        ModelTypeEnum.MLP,
        # ... existing models
        ModelTypeEnum.YOUR_MODEL,  # Add here
    }
    
    if v not in allowed:
        raise NotImplementedError(f"model_type {v} not implemented yet.")
    
    return v
```

#### Step 5: Choose Strategies

Your model must specify:
- **Training Strategy**: How the model learns
  - `EpochTrainingStrategy` - For neural networks with epoch-based training
  - `FitOnceStrategy` - For sklearn-like models with `.fit()` method
  - Custom strategy - If neither fits your needs
  
- **Prediction Strategy**: How the model makes predictions
  - `TorchPredictionStrategy` - For PyTorch models
  - `SklearnPredictionStrategy` - For sklearn-compatible models
  - Custom strategy - For specialized prediction logic

#### Step 6: Add Tests

See [Testing Guidelines](#testing-guidelines) for details.

---

### Creating Training Strategies

Create a new training strategy when:
- Existing strategies don't fit your training paradigm
- You need specialized training loops (e.g., GAN training, meta-learning)
- You require custom callbacks or early stopping logic

#### Requirements

Your training strategy must:

1. **Inherit from `TrainingStrategy`**

2. **Implement the `train()` method** with signature:
   ```python
   def train(
       self,
       model: Any,
       x_train: Any,
       y_train: Any,
       x_val: Any = None,
       y_val: Any = None,
       **kwargs,
   ) -> dict[str, Any]
   ```

3. **Return a dictionary** containing:
   - `"model"` - The trained model instance
   - `"train_loss"` - List of training losses (optional)
   - `"val_loss"` - List of validation losses (optional)

4. **Define properties:**
   - `requires_optimizer` - Boolean indicating if optimizer is needed
   - `requires_criterion` - Boolean indicating if loss function is needed

#### Best Practices

- Use `tqdm` for progress bars
- Handle validation data gracefully (it may be None)
- Log meaningful metrics
- Support early stopping when applicable
- Handle device placement for PyTorch models
- Extract necessary kwargs at the start of `train()`

---

### Creating Prediction Strategies

Create a new prediction strategy when:
- Your model outputs require special post-processing
- You implement ensemble methods
- You need specialized inference logic

#### Requirements

Your prediction strategy must:

1. **Inherit from `PredictionStrategy`**

2. **Implement the `predict()` method** with signature:
   ```python
   def predict(
       self,
       model: Any,
       task: TaskType | None = TaskType.CLASSIFICATION,
       **kwargs,
   ) -> np.ndarray
   ```

3. **Return predictions** as `np.ndarray`

4. **Define the `requires_dataloader()` method:**
   - Return `True` if your strategy needs PyTorch DataLoader
   - Return `False` if it works directly with arrays

#### Best Practices

- Handle both classification and regression tasks
- Support different input formats (DataFrame, array, tensor)
- Set model to evaluation mode for PyTorch
- Disable gradients during inference
- Validate that required inputs are provided
- Convert final output to numpy array

---

## Development Workflow

### Branch Naming Convention

- `feature/model-name` - For new models
- `feature/strategy-name` - For new strategies
- `bugfix/issue-description` - For bug fixes
- `docs/topic` - For documentation updates

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed

Fixes #issue_number
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style changes (formatting)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

**Examples:**
```
feat(models): add XGBoost model implementation

- Add XGBoostConfig with validation
- Implement XGBoostModel wrapper
- Add tests for training and prediction

Closes #42
```

---

## Testing Guidelines

### Required Tests

Every contribution should include:

1. **Unit Tests** for your model/strategy:
   - Model initialization
   - Configuration validation
   - Forward pass (if applicable)
   - Save/load functionality
   - Strategy execution

2. **Integration Tests**:
   - Full training pipeline
   - Prediction pipeline
   - Cross-validation (if supported)

### Running Tests

### Test Structure

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def train(self, model, x_train, y_train, **kwargs):
    """Train the model using provided data.
    
    This method implements the training loop for the model,
    handling batch processing, loss computation, and optimization.
    
    Args:
        model: Model instance to train.
        x_train: Training features.
        y_train: Training labels.
        **kwargs: Additional training parameters:
            - epochs (int): Number of training epochs.
            - batch_size (int): Batch size for training.
            - optimizer: Optimizer instance.
    
    Returns:
        Dictionary containing:
            - model: Trained model instance
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
    
    Raises:
        ValueError: If required parameters are missing.
    
    Example:
        >>> strategy = CustomStrategy()
        >>> history = strategy.train(model, X, y, epochs=10)
        >>> print(f"Final loss: {history['train_loss'][-1]}")
    """
```

### What to Document

- **Classes**: Purpose, attributes, usage examples
- **Methods**: Parameters, return values, side effects, examples
- **Configuration**: All fields with descriptions
- **Validators**: What they check and why

---

## Submitting Pull Requests

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Docstrings are complete and accurate
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow convention

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] New model
- [ ] New strategy
- [ ] Bug fix
- [ ] Documentation update
- [ ] Other (describe)

## Model/Strategy Details
- **Model Name**: [Name]
- **Training Strategy**: [Strategy]
- **Prediction Strategy**: [Strategy]
- **Supported Tasks**: [Classification/Regression/Both]

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests pass locally

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Related issues linked

## Additional Notes
Any additional context or notes for reviewers.
```

### Review Process

1. Automated checks run (tests, linting)
2. Maintainers review code
3. Address review comments
4. Approval and merge

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable names

### Linting and Formatting

We use:
- **Black** for code formatting
- **mypy** for type checking

Run before committing:
```bash
bin/lint
```

---

## Getting Help

### Resources

- **Documentation**: Check `/docs` folder
- **Examples**: See `/examples` folder
- **Tests**: Review existing tests for patterns

### Contact

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: For code contributions

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing! 🎉