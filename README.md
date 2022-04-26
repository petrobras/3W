# Table of Content

* [Introduction](#introduction)
  * [Purpose](#purpose)
  * [Motivation](#motivation)
  * [Strategy](#strategy)
  * [Ambition](#ambition)
  * [Guidelines](#guidelines)
* [3W dataset](#3w-dataset)
  * [Description](#description)
  * [Structure](#structure)
  * [Overview](#overview)
  * [Citation](#citation)
* [3W toolkit](#3w-toolkit)
  * [Description](#description-1)
  * [Examples of use](#examples-of-use)

# Introduction

This is the supporting repository for a Petrobras' project called **3W**, which provides and uses data from ***3*** different sources related to undesirable events that occur in offshore oil ***W***ells during their respective production phases.

[this](discussions/3)

## Purpose

The 3W project intends to promote experimentation of Machine Learning-based approaches and algorithms for specific problems related to undesirable events that occur in offshore oil wells during their respective production phases.

## Motivation

Timely detection of undesirable events in oil wells can help prevent production losses, reduce maintenance costs, environmental accidents, and human casualties. Losses related to this type of events can reach 5% of production in certain scenarios, especially in areas such as Flow Assurance and Artificial Lifting Methods. In terms of maintenance, the cost of a maritime probe, required to perform various types of operations, can exceed US $500,000 per day.

Creating a dataset and making it public to be openly experienced can greatly foment the development of tools that can (i) improve the process of identifying unwanted events in offshore wells production and (ii) increase the efficiency of monitoring the integrity of wells and subsea systems, whose related problems can generate invaluable losses for people, environment, and company's image.

## Strategy

The 3W is an ***open project*** composed by two major resources:

* The [3W dataset](#3w-dataset), which will be evolved and supplemented with more instances from time to time; 
* The [3W toolkit](#3w-toolkit), which will cover an increasing number of undesirable events during its development.

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
* If you have questions about this project, please write an e-mail to `ricardo.vargas at petrobras dot com dot br` and `jeanaraujo at petrobras dot com dot br`.

# 3W dataset

This is the first of the two major components of the 3W project.

## Description

To the best of its authors' knowledge, this is the first realistic and public dataset with rare undesirable real events in oil wells that can be readily used as a benchmark dataset for development of machine learning techniques related to inherent difficulties of actual data.

For more information about the theory behind this dataset, refer to the paper **A realistic and public dataset with rare undesirable real events in oil wells** published in the **Journal of Petroleum Science and Engineering** (link [here](https://doi.org/10.1016/j.petrol.2019.106223)). 

## Structure

The 3W dataset consists of multiple CSV files saved in the [dataset](dataset) directory and structured as follows. 

There are two types of subdirectory: 

* The [folds](dataset/folds) subdirectory holds all 3W dataset configuration files. For each specific project released in the 3W project there will be a file that will specify how and which data must be loaded for training and testing in multiple folds of experimentation. This scheme allows implementation of cross validation and hyperparameter optimization by the 3W toolkit users. In addition, this scheme allows the user to choose some specific characteristics to the desired experiment. For example: whether or not simulated and/or hand-drawn intances should be considered in the training set. It is important to clarify that specifying which instances make up which folds will always be random but fixed in each configuration file. This is considered necessary so that results obtained for the same problem with different approaches can be compared;
* The other subdirectories holds all 3W dataset data files. The subdirectory names are the instances' labels. Each file represents one instance. The filename reveals its source. All files are standardized as follow. There are one observation per line and one series per column. Columns are separated by commas and decimals are separated by periods. The first column contains timestamps, the last one reveals the observations' labels, and the other columns are the Multivariate Time Series (MTS) (i.e. the instance itself).

## Overview

A 3W dataset's general presentation with some quantities and statistics is available in [this](examples/overview.ipynb) Jupyter Notebook.

## Citation

If you use the 3W dataset with any purpose, please cite the aforementioned paper and the 3W dataset itself as follow, respectively:
 
```
@article{VARGAS2019106223,
title = "A realistic and public dataset with rare undesirable real events in oil wells",
journal = "Journal of Petroleum Science and Engineering",
volume = "181",
pages = "106223",
year = "2019",
issn = "0920-4105",
doi = "https://doi.org/10.1016/j.petrol.2019.106223",
url = "http://www.sciencedirect.com/science/article/pii/S0920410519306357",
author = "Ricardo Emanuel Vaz Vargas and Celso José Munaro and Patrick Marques Ciarelli and André Gonçalves Medeiros and Bruno Guberfain do Amaral and Daniel Centurion Barrionuevo and Jean Carlos Dias de Araújo and Jorge Lins Ribeiro and Lucas Pierezan Magalhães",
keywords = "Fault detection and diagnosis, Oil well monitoring, Abnormal event management, Multivariate time series classification",
abstract = "Detection of undesirable events in oil and gas wells can help prevent production losses, environmental accidents, and human casualties and reduce maintenance costs. The scarcity of measurements in such processes is a drawback due to the low reliability of instrumentation in such hostile environments. Another issue is the absence of adequately structured data related to events that should be detected. To contribute to providing a priori knowledge about undesirable events for diagnostic algorithms in offshore naturally flowing wells, this work presents an original and valuable dataset with instances of eight types of undesirable events characterized by eight process variables. Many hours of expert work were required to validate historical instances and to produce simulated and hand-drawn instances that can be useful to distinguish normal and abnormal actual events under different operating conditions. The choices made during this dataset's preparation are described and justified, and specific benchmarks that practitioners and researchers can use together with the published dataset are defined. This work has resulted in two relevant contributions. A challenging public dataset that can be used as a benchmark for the development of (i) machine learning techniques related to inherent difficulties of actual data, and (ii) methods for specific tasks associated with detecting and diagnosing undesirable events in offshore naturally flowing oil and gas wells. The other contribution is the proposal of the defined benchmarks."
}
```
```
Vargas, Ricardo; Munaro, Celso; Ciarelli, Patrick; Medeiros, André; Amaral, Bruno; Barrionuevo, Daniel; Araújo, Jean; Ribeiro, Jorge; Magalhães, Lucas (2019), “Data for: A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells”, Mendeley Data, v1. http://dx.doi.org/10.17632/r7774rwc7v.1
```

As far as we know, the 3W dataset was useful and cited by the works listed below. If you know any other paper, master's degree dissertation or doctoral thesis that cites the 3W dataset, we will be grateful if you let us know by e-mail (`ricardo.vargas at petrobras dot com dot br` and `jeanaraujo at petrobras dot com dot br`).

1. R.E.V. Vargas, C.J. Munaro, P.M. Ciarelli. A methodology for generating datasets for development of anomaly detectors in oil wells based on Artificial Intelligence techniques. I Congresso Brasileiro em Engenharia de Sistemas em Processos. 2019. https://www.ufrgs.br/psebr/wp-content/uploads/2019/04/Abstract_A019_Vargas.pdf.

1. R.E.V. Vargas. Base de dados e benchmarks para prognóstico de anomalias em sistemas de elevação de petróleo. Universidade
Federal do Espírito Santo. Doctoral thesis. 2019. https://github.com/ricardovvargas/3w_dataset/raw/master/docs/doctoral_thesis_ricardo_vargas.pdf.

1. Yan Li, Tingjian Ge, Cindy Chen. Data Stream Event Prediction Based on Timing Knowledge and State Transitions. PVLDB, 13(10): 1779-1792. 2020. http://www.vldb.org/pvldb/vol13/p1779-li.pdf.

1. Tao Lu, Wen Xia, Xiangyu Zou, Qianbin Xia. Adaptively Compressing IoT Data on the Resource-constrained Edge. 3rd {USENIX} Workshop on Hot Topics in Edge Computing (HotEdge 20). 2020. https://www.usenix.org/system/files/hotedge20_paper_lu.pdf.

1. Matheus A. Marins, Bettina D. Barros, Ismael H. Santos, Daniel C. Barrionuevo, Ricardo E.V. Vargas, Thiago de M. Prego, Amaro A. de Lima, Marcello L.R. de Campos, Eduardo A.B. da Silva, Sergio L. Netto. Fault detection and classification in oil wells and production/service lines using random forest. Journal of Petroleum Science and Engineering. 2020. https://doi.org/10.1016/j.petrol.2020.107879.

1. W. Fernandes Junior, R.E.V. Vargas, K.S. Komati, K.A. de Souza Gazolli. Detecção de anomalias em poços produtores de petróleo usando aprendizado de máquina. XXIII Congresso Brasileiro de Automática. 2020. https://www.sba.org.br/open_journal_systems/index.php/cba/article/download/1405/1005.

1. Jiangguo Liu, Jianli Gu, Huishu Li, Kenneth H. Carlson. Machine learning and transport simulations for groundwater anomaly detection,
Journal of Computational and Applied Mathematics. 2020. https://doi.org/10.1016/j.cam.2020.112982.

1. Eduardo S.P. Sobrinho, Felipe L. Oliveira, Jorel L.R. Anjos, Clemente Gonçalves, Marcus V.D. Ferreira, Lucas G.O. Lopes, William W.M. Lira, João P.N. Araújo, Thiago B. Silva, Lucas P. Gouveia. Uma ferramenta para detectar anomalias de produção utilizando aprendizagem profunda e árvore de decisão. Rio Oil & Gas Expo and Conference 2020. 2020. https://icongresso.ibp.itarget.com.br/arquivos/trabalhos_completos/ibp/3/final.IBP0938_20_27112020_085551.pdf.

1. I.M.N. Oliveira. Técnicas de inferência e previsão de dados como suporte à análise de integridade de revestimentos. Universidade Federal de Alagoas. Master's degree dissertation. 2020. https://github.com/ricardovvargas/3w_dataset/raw/master/docs/master_degree_dissertation_igor_oliveira.pdf.

1. Luiz Müller, Marcelo Ramos Martins. Proposition of Reliability-based Methodology for Well Integrity Management During Operational Phase. 30th European Safety and Reliability Conference and 15th Probabilistic Safety Assessment and Management Conference. 2020. https://doi.org/10.3850%2F978-981-14-8593-0_3682-cd.

1. R.S.F. Nascimento, B.H.G. Barbosa, R.E.V. Vargas, I.H.F. Santos. Detecção de falhas com Stacked Autoencoders e técnicas de reconhecimento de padrões em poços de petróleo operados por gas lift. XXIII Congresso Brasileiro de Automática. 2020. https://www.sba.org.br/open_journal_systems/index.php/cba/article/view/1462/1300.

1. R.S.F. Nascimento, B.H.G. Barbosa, R.E.V. Vargas, I.H.F. Santos. Detecção de anomalias em poços de petróleo surgentes com Stacked Autoencoders. Simpósio Brasileiro de Automação Inteligente. 2021. 

1. R.S.F. Nascimento, B.H.G. Barbosa, R.E.V. Vargas, I.H.F. Santos. Fault detection with Stacked Autoencoders and pattern recognition techniques in gas lift operated oil wells. CILAMCE-PANACM. 2021.

1. R.S.F. Nascimento. Detecção de anomalias em poços de produção de petróleo offshore com a utilização de autoencoders e técnicas de reconhecimento de padrões. Universidade Federal de Lavras. Master's degree dissertation. 2021. https://github.com/ricardovvargas/3w_dataset/raw/master/docs/master_degree_dissertation_rodrigo_nascimento.pdf.

1. Taimur Hafeez, Lina Xu, Gavin Mcardle. Edge Intelligence for Data Handling and Predictive Maintenance in IIOT. IEEE Access. 2021. https://ieeexplore.ieee.org/document/9387301.

1. Aurea Soriano-Vargas, Rafael Werneck, Renato Moura, Pedro Mendes Júnior, Raphael Prates, Manuel Castro, Maiara Gonçalves, Manzur Hossain, Marcelo Zampieri, Alexandre Ferreira, Alessandra Davólio, Bernd Hamann, Denis José Schiozer, Anderson Rocha. A visual analytics approach to anomaly detection in hydrocarbon reservoir time series data. Journal of Petroleum Science and Engineering. 2021. https://doi.org/10.1016/j.petrol.2021.108988.

1. Yan Li, Tingjian Ge. Imminence Monitoring of Critical Events: A Representation Learning Approach. International Conference on Management of Data. 2021. https://doi.org/10.1145/3448016.3452804.

1. B.G. Carvalho, R.E.V. Vargas, R.M. Salgado, C.J. Munaro, F.M. Varejão. Flow Instability Detection in Offshore Oil Wells with Multivariate Time Series Machine Learning Classifiers. 30th International Symposium on Industrial Electronics. 2021. https://doi.org/10.1109/ISIE45552.2021.9576310.

1. B.G. Carvalho, R.E.V. Vargas, R.M. Salgado, C.J. Munaro, F.M. Varejão. Hyperparameter Tuning and Feature Selection for Improving Flow Instability Detection in Offshore Oil Wells. IEEE 19th International Conference on Industrial Informatics (INDIN). 2021. https://doi.org/10.1109/INDIN45523.2021.9557415.

1. B.G. Carvalho. Evaluating machine learning techniques for detection of flow instability events in offshore oil wells. Universidade Federal do Espírito Santo. Master's degree dissertation. 2021. https://github.com/ricardovvargas/3w_dataset/raw/master/docs/master_degree_dissertation_bruno_carvalho.pdf.

1. E. M. Turan, J. Jäschke. Classification of undesirable events in oil well operation. 23rd International Conference on Process Control (PC). 2021. https://doi.org/10.1109/PC52310.2021.9447527.

1. I.S. Figueirêdo, T.F. Carvalho, W.J.D Silva, L.L.N. Guarieiro, E.G.S. Nascimento. Detecting Interesting and Anomalous Patterns In Multivariate Time-Series Data in an Offshore Platform Using Unsupervised Learning. OTC Offshore Technology Conference. 2021. https://doi.org/10.4043/31297-MS.

1. R. Karl, J. Takeshita, T. Jung. Cryptonite: A Framework for Flexible Time-Series Secure Aggregation with Non-interactive Fault Recovery. Lecture Notes of the Institute for Computer Sciences, Social-Informatics and Telecommunications Engineering, LNICST. 2021. https://eprint.iacr.org/2020/1561.pdf.

1. A.O. de Salvo Castro, M. de Jesus Rocha Santos, F.R. Leta, C.B.C. Lima, G.B.A. Lima. Unsupervised Methods to Classify Real Data from Offshore Wells. American Journal of  Operations Research. 2021. https://doi.org/10.4236/ajor.2021.115014.

1. W. Fernandes Junior. Comparação de classificadores para detecção de anomalias em poços produtores de petróleo. Instituto Federal do Espírito Santo. Master's degree dissertation. 2022. https://github.com/ricardovvargas/3w_dataset/raw/master/docs/master_degree_dissertation_wander_junior.pdf.

# 3W toolkit

This is the second of the two 3W project's major components.

## Description

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
