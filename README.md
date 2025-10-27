
[![Apache 2.0][apache-shield]][apache] 
[![CC BY 4.0][cc-by-shield]][cc-by]
[![Code style][black-shield]][black]
[![Versioning][semver-shield]][semver]

[apache]: https://opensource.org/licenses/Apache-2.0
[apache-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
[black]: https://github.com/psf/black
[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg
[semver]: https://semver.org
[semver-shield]: https://img.shields.io/badge/semver-2.0.0-blue

# Table of Content

* [Introduction](#introduction)
  * [Motivation](#motivation)
  * [Strategy](#strategy)
  * [Ambition](#ambition)
  * [Governance](#governance)
  * [Contributions](#contributions)
  * [Licenses](#licenses)
  * [Versioning](#versioning)
  * [Questions](#questions)
* [3W Dataset](#3w-dataset)
  * [Structure](#structure)
  * [Overview](#overview)
* [3W Toolkit](#3w-toolkit)
  * [Structure](#structure-1)
  * [Incorporated Problems](#incorporated-problems)
  * [Examples of Use](#examples-of-use)
  * [Reproducibility](#reproducibility)
* [3W Community](#3w-community)

# Introduction

This is the first repository published by Petrobras on GitHub. It supports the 3W Project, which aims to promote experimentation and development of Machine Learning-based approaches and algorithms for specific problems related to detection and classification of undesirable events that occur in offshore oil wells. 
				
The 3W Project is based on the 3W Dataset, a database described in [this paper](https://doi.org/10.1016/j.petrol.2019.106223), and on the 3W Toolkit, a software package that promotes experimentation with the 3W Dataset for specific problems. The name **3W** was chosen because this dataset is composed of instances from ***3*** different sources and which contain undesirable events that occur in oil ***W***ells.

## Motivation

Timely detection of undesirable events in oil wells can help prevent production losses, reduce maintenance costs, environmental accidents, and human casualties. Losses related to this type of events can reach 5% of production in certain scenarios, especially in areas such as Flow Assurance and Artificial Lifting Methods. In terms of maintenance, the cost of a maritime probe, required to perform various types of operations, can exceed US $500,000 per day.

Creating a dataset and making it public to be openly experienced can greatly foment the development of tools that can:

* Improve the process of identifying undesirable events in the drilling, completion and production phases of offshore wells;
* Increase the efficiency of monitoring the integrity of wells and subsea systems, whose related problems can generate invaluable losses for people, environment, and company's image.

## Strategy

The 3W is the first pilot of a Petrobras' program called [Conexões para Inovação - Módulo Open Lab](https://tecnologia.petrobras.com.br/modulo-open-lab). This pilot is an ***open project*** composed by two major resources:

* The [3W Dataset](#3w-dataset), which will be evolved and supplemented with more instances from time to time; 
* The [3W Toolkit](#3w-toolkit), which will also be evolved (in many ways) to cover an increasing number of undesirable events during its development.

Therefore, our strategy is to make these resources publicly available so that we can develop the 3W Project with a global community collaboratively.

## Ambition

With this project, Petrobras intends to develop (fix, improve, supplement, etc.):

* The [3W Dataset](#3w-dataset) itself;
* The [3W Toolkit](#3w-toolkit) itself;
* Approaches and algorithms that can be incorporated into systems dedicated to monitoring undesirable events in offshore oil wells during their respective drilling, completion and production phases;
* Tools that can be useful for our ambition.

## Governance

The 3W Project was conceived and publicly launched on May 30, 2022 as a strategic action by Petrobras, led by its department responsible for Flow Assurance and its research center ([CENPES](https://www.petrobras.com.br/inovacao-e-tecnologia/centro-de-pesquisa)). Since then, 3W has become increasingly consolidated at Petrobras in several aspects: more professionals specialized in labeling instances, more projects and teams using the resources made available by 3W, more investment in developing the digital tools needed to label and export instances, more interest in including different types of undesirable events that occur in wells during the drilling, completion and production phases, etc. 

Due to this evolution, from May 1st, 2024 the 3W's governance is now done with the participation of the Petrobras' department responsible for Well Integrity.

## Contributions

We expect to receive various types of contributions from individuals, research institutions, startups, companies and partner oil operators.

Before you can contribute to this project, you need to read and agree to the following documents:

* [CODE OF CONDUCT](CODE_OF_CONDUCT.md);
* [CONTRIBUTOR LICENSE AGREEMENT](CONTRIBUTOR_LICENSE_AGREEMENT.md);
* [CONTRIBUTING GUIDE](CONTRIBUTING.md).

It is also very important to know, participate and follow the discussions. See the discussions section.

## Licenses

All the code of this project is licensed under the [Apache 2.0 License][apache] and all 3W Dataset's data files (Parquet files saved in subdirectories of the [dataset](dataset) directory) are licensed under the [Creative Commons Attribution 4.0 International License][cc-by].

## Versioning

In the 3W Project, three types of versions will be managed as follows.

* Version of the 3W Toolkit: specified in the [__init__.py](toolkit/__init__.py) file;
* Version of the 3W Dataset: specified in the [dataset.ini](dataset/dataset.ini) file;
* Version of the 3W Project: specified with tags in the git repository;
* We will exclusively use the semantic versioning defined in https://semver.org;
* Versions will always be updated manually;
* Versioning of the 3W Toolkit and 3W Dataset are completely independent of each other;
* The version of the 3W Project will be updated whenever, and only when, there is a new commit in the `main` branch of the repository, regardless of the updated resource: 3W Toolkit, 3W Dataset, 3W Project's documentation, example of use, etc;
* We will only use annotated tags and for each tag there will be a release in the remote repository (GitHub);
* Content for each release will be automatically generated with functionality provided by GitHub.

## Questions

See the discussions section. If you don't get clarification, please open discussions to ask your questions so we can answer them.

# 3W Dataset

To the best of its authors' knowledge, this is the first realistic and public dataset with rare undesirable real events in oil wells that can be readily used as a benchmark dataset for development of machine learning techniques related to inherent difficulties of actual data. For more information about the theory behind this dataset, refer to the paper **A realistic and public dataset with rare undesirable real events in oil wells** published in the **Journal of Petroleum Science and Engineering** (link [here](https://doi.org/10.1016/j.petrol.2019.106223)). 

## Structure

The 3W Dataset consists of multiple Parquet files saved in subdirectories of the [dataset](dataset) directory and structured as detailed [here](3W_DATASET_STRUCTURE.md). 

## Overview

A 3W Dataset's general presentation with some quantities and statistics is available in [this](overviews/_baseline/main.ipynb) Jupyter Notebook.

# 3W Toolkit

The 3W Toolkit is a software package written in Python 3 that contains resources that make the following easier:

* [3W Dataset](#3w-dataset) overview generation;
* Experimentation and comparative analysis of Machine Learning-based approaches and algorithms for specific problems related to undesirable events that occur in offshore oil wells during their respective drilling, completion and production phases;
* Standardization of key points of the Machine Learning-based algorithm development pipeline.

It is important to note that there are arbitrary choices in this toolkit, but they have been carefully made to allow adequate comparative analysis without compromising the ability to experiment with different approaches and algorithms.

## Structure

The 3W Toolkit is implemented in sub-modules as discribed [here](3W_TOOLKIT_STRUCTURE.md).

## Incorporated Problems

Specific problems will be incorporated into this project gradually. At this point, we can work on:

* [Binary classifier of Spurious Closure of DHSV](problems/01_binary_classifier_of_spurious_closure_of_dhsv/README.md).

All specification is detailed in the [CONTRIBUTING GUIDE](CONTRIBUTING.md).

## Examples of Use

The list below with examples of how to use the 3W Toolkit will be incremented throughout its development.

* 3W Dataset's overviews:
  * [Baseline](overviews/_baseline/main.ipynb)
  * [André Machado's overview](overviews/AndreMachado/main.ipynb)
* Binary classifier of Spurious Closure of DHSV:
  * [Baseline](problems/01_binary_classifier_of_spurious_closure_of_dhsv/_baseline/main.ipynb)

For a contribution of yours to be listed here, follow the instructions detailed in the [CONTRIBUTING GUIDE](CONTRIBUTING.md).

## Reproducibility

For all results generated by the 3W Toolkit to be consistent, we recommend you create and use a virtual environment with the packages versions specified in the [environment.yml](environment.yml), which was generated with [conda](https://docs.conda.io). Our current recommendation is to use the conda distributed by [Miniforge](https://conda-forge.org/download/). Download and install Miniforge according to the official instructions. Open a prompt on your operating system (Windows, Linux or MacOS). Make sure the current directory is the directory where you have the 3W. Run the following commands as needed:

* To create a virtual environment from our [environment.yml](environment.yml): 
```
$ conda env create -f environment.yml
```
* To activate the created virtual environment:
```
$ conda activate 3W
```
* To use the 3W Toolkit resources interactively:
```
$ python
```
* To initialize a local Jupyter Notebook server:
```
$ jupyter notebook
```

# 3W Community

The 3W Community is gradually expanding and is made up of independent professionals and representatives of research institutions, startups, companies and oil operators from different countries.

More information about this community can be found [here](community/README.md).