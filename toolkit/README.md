[![Apache 2.0][apache-shield]][apache] 
[![Code style][black-shield]][black]
[![Versioning][semver-shield]][semver]

[apache]: https://opensource.org/licenses/Apache-2.0
[apache-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[black]: https://github.com/psf/black
[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg
[semver]: https://semver.org
[semver-shield]: https://img.shields.io/badge/semver-2.0.0-blue

# Table of Content

* [Introduction](#introduction)
* [Release Notes](#release-notes)
  * [1.0.0](#100)
  * [1.1.0](#110)
  * [1.2.0](#120)  

# Introduction

All the 3W Toolkit's sub-modules developed as part of the 3W Project are licensed under the [Apache 2.0 License][apache].

# Release Notes

Each subsection below contains release notes for a specific 3W Toolkit version. Differences from the immediately previous version are highlighted.

## 1.0.0

Release: May 30, 2022.

This was the first published version.

## 1.1.0

Release: July 25, 2024.

Highlights:

1. Makes resources (functions and constants) compatible with 3W Dataset version 2.0.0, which is based on Parquet files.

## 1.2.0

Release: October 19, 2024 # Lastiest version

Highlights:

1. **Adapts `load_dataset()` to 3W Dataset 2.0:** The `load_dataset()` function in `base.py` was adapted to correctly handle the folder structure and different data types of the 3W Dataset 2.0. It was renamed to `load_3w_dataset()`.
2. **Updates `dev.py` for 3W Dataset 2.0:** The `dev.py` sub-module was updated to ensure compatibility with the new `load_3w_dataset()` function and the 3W Dataset 2.0 structure. The `extrai_arrays()` function was removed, and the `EventFolds` and `Experiment` classes were adjusted.
3. **Updates `misc.py` for 3W Dataset 2.0:** The `misc.py` sub-module was updated to ensure compatibility with the new `load_3w_dataset()` function and the 3W Dataset 2.0 structure. Redundant functions were removed, and existing functions were adapted to receive the DataFrame as a parameter.
4. **Updates `__init__.py` for 3W Dataset 2.0:**  The `__init__.py` file was updated to import and expose the new `load_3w_dataset()` function.

These updates ensure that the 3W Toolkit is fully compatible with the 3W Dataset 2.0, providing a more efficient and streamlined workflow for loading and analyzing the data.