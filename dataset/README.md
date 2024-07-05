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

# Introduction

All 3W Dataset data files (CSV files in the subdirectories of the [dataset](dataset) directory) are licensed under the [Creative Commons Attribution 4.0 International License][cc-by].

# Release Notes

Each subsection below contains release notes for a specific 3W Dataset version. Differences from the immediately previous version are highlighted.

## 1.0.0

Release: July 1, 2019.

This was the first published version, which is fully described in [this](https://doi.org/10.1016/j.petrol.2019.106223) paper.

## 1.1.0

Release: December 30, 2022.

Highlights:

1. New instances have been added as follows:
    * 1 instance of event type 7.
1. Instances have been removed due to issues identified as follows:
	* 3 instances of event type 0;
	* 1 instance of event type 5;
	* 3 instances of event type 8, when compared to what is described in the paper **A realistic and public dataset with rare undesirable real events in oil wells** published in the **Journal of Petroleum Science and Engineering** (link [here](https://doi.org/10.1016/j.petrol.2019.106223)).
1. Normal periods of certain instances with anomalies have been increased as possible. We tried to have instances with minimum normal periods of 1 hour;
1. Names of certain files with instances have changed due to increased normal periods;
1. Periods of certain instances have been relabeled;
1. Time series of certain variables were added because these variables were contextualized after the previous version of the 3W Dataset was created;
1. Time series of certain variables were removed because these variables were decontextualized after the previous version of the 3W Dataset was created;
1. Time series of certain variables have been completely changed due to these variables having been recontextualized after the creation of the previous version of the 3W Dataset;
1. Certain variable values ​​have undergone minimal change due to different rounding;
1. The 3W Dataset's main configuration file ([dataset.ini](dataset.ini)) has been updated;
1. The Jupyter Notebook with the [3W Dataset's baseline general presentation](../overviews/_baseline/main.ipynb) has been updated.

## 1.1.1

Release: April 09, 2023.

Highlights:

1. Issue #60 has been resolved;
1. Issue #65 has been resolved;
1. Certain variable values ​​have undergone minimal change due to different rounding;
1. The 3W Dataset's main configuration file ([dataset.ini](dataset.ini)) has been updated;
1. The Jupyter Notebook with the [3W Dataset's baseline general presentation](../overviews/_baseline/main.ipynb) has been updated.