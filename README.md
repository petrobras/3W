# 3W ToolKit

<!-- ref: https://github.com/othneildrew/Best-README-Template/tree/main -->


<!-- label-->
<a id="readme-top"></a>

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url] -->
<!-- [![Unlicense License][license-shield]][license-url] -->

[![Code Coverage](https://img.shields.io/codecov/c/github/Mathtzt/3w_toolkit)](https://codecov.io/github/Mathtzt/3w_toolkit)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)
![version](https://img.shields.io/badge/version-0.1.0-blue)


<!-- TABLE OF CONTENTS -->

<summary>Table of Contents</summary>
<ol>
<li><a href="#about-the-project">About</a></li>
<li><a href="#documentation">Development Documentation</a></li>
<ul>
<li><a href="#dataset_versions">Dataset Versions</a></li>
<li><a href="#data_loader">Data Loader</a></li>
<li><a href="#model_dev">Model Development</a></li>
<li><a href="#model_assessment">Model Assessment</a></li>
</ul>

<li><a href="#documentation2">Usage Documentation</a></li>
<ul>
<li><a href="#toolkit_examples">Toolkit Examples</a></li>
<li><a href="#toolkit_demos">Toolkit Demos</a></li>
<li><a href="#toolkit_challenges">Toolkit Challenges</a></li>
<li><a href="#toolkit_videos">Toolkit Videos</a></li>
</ul>
<li><a href="#uml">Toolkit UML</a></li>

<li><a href="#setup">Setup</a></li>
<li><a href="#requirements">Requiriments</a></li>
<li><a href="#install">Install</a></li>
<li><a href="#contributing">Contributing</a></li>



<li><a href="#license">License</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

---

## About <a id="about-the-project"></a>

The evolution of machine learning has been catalyzed by the rapid advancement in data acquisition systems, scalable storage, high-performance processing, and increasingly efficient model training through matrix-centric hardware (e.g., GPUs). These advances have enabled the deployment of highly parameterized AI models in real-world applications such as health care, finance, and industrial operations.


In the oil & gas sector, the widespread availability of low-cost sensors has driven a paradigm shift from reactive maintenance to condition-based monitoring (CBM), where faults are detected and classified during ongoing operation. This approach minimizes downtime and improves operational safety. The synergy between AI and big data analysis has thus enabled the development of generalizable classifiers that require minimal domain knowledge and can be effectively adapted to a wide range of operational scenarios.
In this context, we present 3WToolkit+, a modular and open-source AI toolkit for time-series processing, aimed at fault detection and classification in oil well operation. Building upon the experience with the original 3WToolkit system and leveraging the Petrobras <a href="https://github.com/petrobras/3W">3W Dataset </a>, 3WToolkit+ introduces enhanced functionalities, such as advanced data imputation, deep feature extraction, synthetic data augmentation, and high-performance computing capabilities for model training.


<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img width="400" src="https://coppe.ufrj.br/wp-content/uploads/2023/10/COPPE-horiz-completa-cor-300dpi.jpg">
    <img width="600" src="https://sites.ufpe.br/litpeg/wp-content/uploads/sites/10/2022/06/Petrobras-Logo.png">
</div>


The development of the 3WToolkit+ is the result of a collaborative partnership between Petrobras, with a focus on the CENPES research center, and the COPPE/Universidade Federal do Rio de Janeiro (UFRJ). This joint effort brings together complementary strengths: COPPE/UFRJ contributes decades of proven expertise in signal processing and machine learning model development, while CENPES offers access to highly specialized technical knowledge and real-world operational challenges in the oil and gas sector. This synergy ensures that 3WToolkit+ is both scientifically rigorous and practically relevant, addressing complex scenarios with robust and scalable AI-based solutions for time-series analysis and fault detection in oil well operations.


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

## Documentation <a id="documentation"></a>

<img width="1200" src="https://github.com/Mathtzt/3w_toolkit/blob/1fa7246adfabfea07007aeae9e406f8bcd282aa6/figures/3w_top_view.png?raw=1">

The image above illustrates the high-level architecture of the 3WToolkit+, designed to support the full pipeline of machine learning applications using the 3W dataset—from raw data ingestion to model evaluation and delivery to end users. Each block in the architecture is briefly described below:

### 3W Dataset Versions <a id="dataset_versions"></a>
This block represents different available versions of the 3W dataset, which include real and simulated data from offshore oil wells. These datasets serve as the foundation for all subsequent stages of data processing, modeling, and evaluation.

### Data Loader <a id="data_loader"></a>
The Data Loader module is responsible for importing, validating, and preparing the raw 3W data for use in model training and evaluation. It handles missing data, standardizes variable formats, and performs initial quality checks to ensure compatibility across toolkit components.

<img width="1200" src="https://github.com/Mathtzt/3w_toolkit/blob/a5673d622924d5b7e2e6ca0d52a1a9719be62683/figures/3w_data_loader.png?raw=1">

### Model Development <a id="model_dev"></a>
This central module provides the infrastructure for designing, training, and optimizing machine learning models for fault detection and classification. It supports both classical and deep learning models and includes tools for hyperparameter tuning, cross-validation, and model versioning.

<img width="1200" src="https://github.com/Mathtzt/3w_toolkit/blob/a5673d622924d5b7e2e6ca0d52a1a9719be62683/figures/3w_model_dev.png?raw=1">

### Assessment <a id="model_assessment"></a>
The Assessment module evaluates model performance using both sample-level and event-level metrics. It includes support for traditional indicators (e.g., accuracy, precision, recall) as well as domain-specific metrics such as detection lag and anticipation time, which are critical for condition-based monitoring.

<img width="1200" src="https://github.com/Mathtzt/3w_toolkit/blob/a5673d622924d5b7e2e6ca0d52a1a9719be62683/figures/3w_assessment.png?raw=1">

## Usage Documentation <a id="documentation2"></a>

### 3W Examples <a id="toolkit_examples"></a>
A curated set of ready-to-use model configurations and scripts that demonstrate how to apply the toolkit to common fault detection tasks using the 3W dataset. These examples accelerate onboarding and reproducibility.

### 3W Tutorials/Demos <a id="toolkit_demos"></a>
Step-by-step tutorials and demonstration notebooks that guide users through the toolkit’s functionalities, explaining how each module operates and how to configure different experiments.

### 3W Challenges <a id="toolkit_challenges"></a>
This component provides benchmarking tasks and open challenges using real scenarios derived from the 3W dataset. It promotes collaborative development and comparative evaluation of machine learning solutions in fault diagnosis.

### 3W Videos <a id="toolkit_videos"></a>
Instructional videos that explain toolkit concepts, walk through complete modeling pipelines, and offer insights from domain experts. These videos aim to broaden accessibility and support training initiatives.


## Toolkit UML <a id="uml"></a>

Building upon the high-level block diagram architecture, a detailed UML (Unified Modeling Language) diagram was developed to support the software engineering and implementation of the 3WToolkit+. The UML model formalizes the relationships between components, data structures, and workflows described in the block-level architecture, enabling a structured and maintainable development process.

This transition from conceptual blocks to formal UML design ensures that each module—such as the Data Loader, Model Development, and Assessment—has clearly defined interfaces, class responsibilities, and interaction protocols. It also facilitates modular programming, unit testing, and future extensibility of the toolkit by providing developers with a shared, consistent blueprint for implementation.

The UML diagram serves not only as an internal reference for the development team but also as part of the developer-oriented documentation that accompanies the toolkit and it is shown bellow

<img width="1200" src="https://github.com/Mathtzt/3w_toolkit/blob/a5673d622924d5b7e2e6ca0d52a1a9719be62683/figures/3w_toolkit_uml.png?raw=1">


## Toolkit Setup  <a id="setup"></a>

### Requirements  <a id="requirements"></a>

### Installation  <a id="install"></a>

## Contributing  <a id="contributing"></a>

## Licenses  <a id="license"></a>


<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: images/screenshot.png

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Mathtzt/3w_toolkit.svg?style=for-the-badge
[contributors-url]: https://github.com/Mathtzt/3w_toolkit/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/Mathtzt/3w_toolkit.svg?style=for-the-badge
[forks-url]: https://github.com/Mathtzt/3w_toolkit/network/members

[stars-shield]: https://img.shields.io/github/stars/Mathtzt/3w_toolkit.svg?style=for-the-badge
[stars-url]: https://github.com/Mathtzt/3w_toolkit/stargazers

[issues-shield]: https://img.shields.io/github/issues/Mathtzt/3w_toolkit.svg?style=for-the-badge
[issues-url]: https://github.com/Mathtzt/3w_toolkit/issues

[license-shield]: https://img.shields.io/github/license/Mathtzt/3w_toolkit.svg?style=for-the-badge
[license-url]: https://github.com/Mathtzt/3w_toolkit/blob/master/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/natanael-moura-junior-425a3294

<!-- 
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com -->