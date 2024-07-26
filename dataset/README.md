[![CC BY 4.0][cc-by-shield]][cc-by]
[![Versioning][semver-shield]][semver]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
[semver]: https://semver.org
[semver-shield]: https://img.shields.io/badge/semver-2.0.0-blue

# Table of Content

* [Introduction](#introduction)
* [Release Notes](#release-notes)
  * [1.0.0](#100)
  * [1.1.0](#110)
  * [1.1.1](#111)
  * [2.0.0](#200)

# Introduction

All 3W Dataset data files (files in the subdirectories of the [dataset](dataset) directory) are licensed under the [Creative Commons Attribution 4.0 International License][cc-by].

# Release Notes

Each subsection below contains release notes for a specific 3W Dataset version. Differences from the immediately previous version are highlighted.

## 1.0.0

Release: July 1, 2019.

This was the first published version, which is fully described in [this](https://doi.org/10.1016/j.petrol.2019.106223) paper.

## 1.1.0

Release: December 30, 2022.

Highlights:

1. New instances were added as follows:
    * 1 instance of event type 7.
1. Instances were removed due to issues identified as follows:
	* 3 instances of event type 0;
	* 1 instance of event type 5;
	* 3 instances of event type 8, when compared to what is described in the paper **A realistic and public dataset with rare undesirable real events in oil wells** published in the **Journal of Petroleum Science and Engineering** (link [here](https://doi.org/10.1016/j.petrol.2019.106223)).
1. Normal periods of certain instances with anomalies were increased as possible. We tried to have instances with minimum normal periods of 1 hour;
1. Names of certain files with instances have changed due to increased normal periods;
1. Labels in some real instances were adjusted by experts;
1. All values of some variables in some real instances were corrected due to corrections in historian systems' tag configurations;
1. Certain variable values ​​have undergone minimal change due to different rounding;
1. The 3W Dataset's main configuration file ([dataset.ini](dataset.ini)) was updated.

## 1.1.1

Release: April 09, 2023.

Highlights:

1. Issue #60 was resolved;
1. Issue #65 was resolved;
1. Certain variable values ​​have undergone minimal change due to different rounding;
1. The 3W Dataset's main configuration file ([dataset.ini](dataset.ini)) was updated.

## 2.0.0

Release: July 25, 2024.

Highlights:

1. All instances are now saved in Parquet files (created with the `pyarrow` engine and `brotli` compression);
1. Reduction in disk space occupied by the 3W Dataset of 3.15 GB (from 4.89 GB to 1.74 GB);
1. Real and simulated instances of type 9 were added;
1. Several instances of types 0, 3, 4, 5, 6 and 8 were added;
1. Another 24 real wells were covered with new real instances (now 42 real wells are covered);
1. Some real instances, mainly of type 1, were removed;
1. 1 variable was removed (`T-JUS-CKGL`);
1. Another 20 variables were added (there are now 27 variables);
1. Another label referring to well operational status was added;
1. Normal periods in several real instances with unwanted events were extended;
1. All labeling gaps in real instances were eliminated (all observations were labeled);
1. Conversions between measurement units in several instances were corrected;
1. Labels in several real instances were adjusted by experts;
1. All values of some variables in some real instances were corrected due to corrections in historian systems' tag configurations;
1. Certain variable values ​​have undergone minimal change due to different rounding;
1. The 3W Dataset's main configuration file ([dataset.ini](dataset.ini)) was updated.