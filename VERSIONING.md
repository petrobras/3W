# Versioning in the 3W Project

This document describes the versioning strategy adopted by the 3W Project, including where each version is specified, how it is managed, and, for the 3W Toolkit, the detailed rules for Semantic Versioning ([SemVer](https://semver.org/)).

## Overview of the Three Version Types

The 3W Project manages three independent version types:

| Version Type      | Location                                      | Description                                      |
|-------------------|-----------------------------------------------|--------------------------------------------------|
| **3W Toolkit**    | [pyproject.toml](pyproject.toml)               | AI toolkit for time-series processing            |
| **3W Dataset**    | [dataset/dataset.ini](dataset/dataset.ini)     | Parquet data and configuration                  |
| **3W Project**     | [Git annotated tags in the repository](https://github.com/petrobras/3W/tags)          | Overall project releases                         |

### General Rules

* All versions follow the [Semantic Versioning](https://semver.org) specification (SemVer).
* Versions are updated **manually**.
* The **3W Toolkit** and **3W Dataset** versions are **independent** of each other.
* The **3W Project** version is updated whenever, and only when, there is a new commit in the `main` branch, regardless of which resource changed (**Toolkit**, **Dataset**, **documentation**, **examples**, etc.).
* Release content is generated automatically using GitHub functionality.

---

## 3W Toolkit Versioning (SemVer)

The 3W Toolkit uses [Semantic Versioning 2.0.0](https://semver.org). Version numbers take the form:

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

### What "API" Means in This Document

In the context of the 3W Toolkit, **API** refers to the **public programming interface**, the surface that other developers use when they import and call the toolkit. For this toolkit, the API includes:

* **Public classes and functions** that users import (e.g., `PlotMultipleSeries`, `CorrelationHeatmap`);
* **Function and constructor signatures** (parameters, their order, names, and whether they are required or optional);
* **Return types and behavior** of public methods;
* **Public attributes and methods** exposed by classes.

When we speak of "incompatible API changes" or "backward-compatible additions," we mean changes to this programming interface.

### The Three Increments

#### MAJOR version (X.y.z)

**Increment when:** You make **incompatible API changes**.

Incompatible changes break existing code that depends on the public API. Users upgrading from a previous MAJOR version may need to modify their code.

| Situation | Example |
|-----------|---------|
| **Removing a public function or class** | Removing `PlotMultipleSeries` or renaming it in a way that breaks imports. |
| **Changing function signatures** | Changing `plot_multiple_series(series_list, labels)` to `plot_multiple_series(data, labels, title)` and removing `series_list`. |
| **Changing return types** | A function that returned `tuple[Figure, Axes]` now returns only `Axes`. |
| **Changing behavior in a breaking way** | A model that previously accepted `n_features` as first positional argument now requires it as a keyword-only argument. |
| **Removing or renaming parameters** | Dropping the `xlabel` parameter from a visualizer constructor. |
| **Deprecation removal** | Removing a function that was deprecated in a previous MINOR release. |

**Example:** `2.1.0` → `3.0.0`

---

#### MINOR version (x.Y.z)

**Increment when:** You add **functionality in a backward-compatible manner**.

New features are added without breaking existing code. Existing callers continue to work without changes.

| Situation | Example |
|-----------|---------|
| **Adding new functions or classes** | Adding `PlotFeatureImportance` or a new preprocessing step. |
| **Adding optional parameters** | Adding `figsize=(12, 6)` with a default value to an existing function. |
| **Adding new methods to a class** | Adding a `to_dict()` method to a model class. |
| **Extending supported inputs** | A loader that accepted only Parquet now also accepts CSV. |
| **Deprecating functionality** | Marking a deprecated parameter with a warning; the old behavior still works. |
| **Improving performance** | Internal optimizations that **do not change the public API**. |

**Example:** `2.1.0` → `2.2.0`

---

#### PATCH version (x.y.Z)

**Increment when:** You make **backward-compatible bug fixes**.

Fixes correct incorrect behavior without changing the public API or adding new features.

| Situation | Example |
|-----------|---------|
| **Fixing incorrect output** | A metric that returned wrong values for edge cases now returns correct values. |
| **Fixing crashes** | Handling `None` or empty inputs that previously caused exceptions. |
| **Fixing documentation** | Correcting misleading docstrings or examples. |
| **Fixing type hints** | Correcting incorrect type annotations without changing runtime behavior. |
| **Security fixes** | Addressing vulnerabilities that do not change the API contract. |

**Example:** `2.1.0` → `2.1.1`

---

### When a Release Contains Multiple Types of Changes

If a single release (e.g., one PR or one commit) includes changes of different impact levels, use the **highest** increment:

* Patch + Minor → bump **MINOR**
* Patch + Major → bump **MAJOR**
* Minor + Major → bump **MAJOR**

This ensures the version number reflects the most impactful change and avoids under-reporting breaking changes.

---

### Pre-release and Build Metadata

SemVer allows optional suffixes:

* **Pre-release:** `-alpha`, `-beta`, `-rc.1` (e.g., `2.2.0-alpha`, `2.2.0-rc.1`)
* **Build metadata:** `+20240226`, `+sha.5114f85` (e.g., `2.2.0+20240226`)

| Format | Use case |
|--------|----------|
| `2.2.0-alpha` | Early testing; API may still change. |
| `2.2.0-beta.1` | Feature-complete; final testing before release. |
| `2.2.0-rc.1` | Release candidate; intended to match final 2.2.0. |
| `2.2.0+20240226` | Build metadata; does not affect version precedence. |

Pre-release versions have **lower precedence** than the corresponding release: `2.2.0-alpha` < `2.2.0`.

---

### Version Precedence

When comparing versions:

* `1.0.0` < `2.0.0` < `2.1.0` < `2.1.1`
* `1.0.0-alpha` < `1.0.0-beta` < `1.0.0-rc.1` < `1.0.0`

---

### Resetting Lower Components

When incrementing a higher component, reset lower components to zero:

* MAJOR bump: `2.1.3` → `3.0.0`
* MINOR bump: `2.1.3` → `2.2.0`
* PATCH bump: `2.1.3` → `2.1.4`

---

## References

* [Semantic Versioning 2.0.0](https://semver.org) — Full specification
* [Python Packaging: Versioning](https://packaging.python.org/en/latest/discussions/versioning/) — Python-specific guidance
* [README — Versioning](README.md#versioning) — Summary of version locations and rules
