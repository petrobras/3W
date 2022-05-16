# Table of Content

* [Introduction](#introduction)
  * [Purpose](#purpose)
  * [Motivation](#motivation)
  * [Strategy](#strategy)
  * [Ambition](#ambition)
  * [Guidelines](#guidelines)
* [3W dataset](#3w-dataset)
  * [Structure](#structure)
  * [Overview](#overview)
  * [Citation](#citation)
* [3W toolkit](#3w-toolkit)
  * [Examples of use](#examples-of-use)

# Introduction

This is the supporting repository for a Petrobras' project called **3W**, which provides and uses data from ***3*** different sources related to undesirable events that occur in offshore oil ***W***ells during their respective production phases.

## Purpose

The 3W project intends to promote experimentation of Machine Learning-based approaches and algorithms for specific problems related to undesirable events that occur in offshore oil wells during their respective production phases.

## Motivation

Timely detection of undesirable events in oil wells can help prevent production losses, reduce maintenance costs, environmental accidents, and human casualties. Losses related to this type of events can reach 5% of production in certain scenarios, especially in areas such as Flow Assurance and Artificial Lifting Methods. In terms of maintenance, the cost of a maritime probe, required to perform various types of operations, can exceed US $500,000 per day.

Creating a dataset and making it public to be openly experienced can greatly foment the development of tools that can:

* Improve the process of identifying unwanted events in offshore wells production;
* Increase the efficiency of monitoring the integrity of wells and subsea systems, whose related problems can generate invaluable losses for people, environment, and company's image.

## Strategy

The 3W is an ***open project*** composed by two major resources:

* The [3W dataset](#3w-dataset), which will be evolved and supplemented with more instances from time to time; 
* The [3W toolkit](#3w-toolkit), which will also be evolvee to cover an increasing number of undesirable events during its development.

## Ambition

With this project, Petrobras intends to develop (fix, improve, supplement, etc.):

* The [3W dataset](#3w-dataset) itself;
* The [3W toolkit](#3w-toolkit) itself;
* Algorithms that can be incorporated into systems dedicated to monitoring undesirable events in offshore oil wells during their respective production phases.

## Guidelines

* All resources made available in this repository are licensed under the terms defined in [this](LICENSE) document;
* If you wish to contribute to this project, please read the following documents:
  * [CONTRIBUTOR LICENSE AGREEMENT](CONTRIBUTOR_LICENSE_AGREEMENT.md);
  * [CONTRIBUTING GUIDELINES](CONTRIBUTING_GUIDELINES.md);
  * [CODE OF CONDUCT](CODE_OF_CONDUCT.md).
* If you have questions, see the discussions section. If you don't get clarification from this feature, please open discussions to ask yours questions so we can answer them.

# 3W dataset

To the best of its authors' knowledge, this is the first realistic and public dataset with rare undesirable real events in oil wells that can be readily used as a benchmark dataset for development of machine learning techniques related to inherent difficulties of actual data. For more information about the theory behind this dataset, refer to the paper **A realistic and public dataset with rare undesirable real events in oil wells** published in the **Journal of Petroleum Science and Engineering** (link [here](https://doi.org/10.1016/j.petrol.2019.106223)). 

## Structure

The 3W dataset consists of multiple CSV files saved in the [dataset](dataset) directory and structured as detailed [here](3W_DATASET_STRUCTURE.md). 

## Overview

A 3W dataset's general presentation with some quantities and statistics is available in [this](examples/overview.ipynb) Jupyter Notebook.

## Citation

As far as we know, the 3W dataset was useful and cited by the works listed [here](CITATIONS.md). If you know any other paper, master's degree dissertation or doctoral thesis that cites the 3W dataset, we will be grateful if you let us know by commenting [this](https://github.com/Petrobras/3W/discussions/3) discussion. If you use the 3W dataset with any purpose, please cite the aforementioned paper and the 3W dataset itself as specified [here](CITE.md).

# 3W toolkit

The 3W toolkit is a software package written in Python 3 that contains resources that make the following easier:

* [3W dataset](#3w-dataset) overview generation;
* Standardization of key points of the Machine Learning-based algorithm development pipeline;
* Experimentation and comparative analysis of approaches and algorithms for specific problems related to undesirable events that occur in offshore oil wells during their respective production phases.

Specific problems will be incorporated into this project gradually. At this time, models can be developed for the following problems:

* Binary Classifier of Spurious Closure of DHSV.

It is important to note that there are arbitrary choices in this toolkit, but they have been carefully made to allow adequate comparative analysis without compromising the ability to experiment with different approaches and algorithms.

## Examples of use

The list below with examples of how to use this toolkit will be incremented throughout its development.

* [3W dataset's overview](examples/overview.ipynb)
* [Binary Classifier of Spurious Closure of DHSV](examples/binary_classifier_of_spurious_closure_of_dhsv.ipynb)