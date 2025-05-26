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
    <ul>
        <li><a href="#docker">Docker</a></li>
        <li><a href="#vscode_docker">Development in VSCode using Docker</a></li>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#install">Installation</a></li>
    </ul>

<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgments">Acknowledgments</a></li>
</ol>

---

## About <a id="about-the-project"></a>

<p style="text-align: justify;">The evolution of machine learning has been catalyzed by the rapid advancement in data acquisition systems, scalable storage, high-performance processing, and increasingly efficient model training through matrix-centric hardware (e.g., GPUs). These advances have enabled the deployment of highly parameterized AI models in real-world applications such as health care, finance, and industrial operations.</p>

<p style="text-align: justify;">In the oil & gas sector, the widespread availability of low-cost sensors has driven a paradigm shift from reactive maintenance to condition-based monitoring (CBM), where faults are detected and classified during ongoing operation. This approach minimizes downtime and improves operational safety. The synergy between AI and big data analysis has thus enabled the development of generalizable classifiers that require minimal domain knowledge and can be effectively adapted to a wide range of operational scenarios.</p>

<p style="text-align: justify;">In this context, we present 3WToolkit+, a modular and open-source AI toolkit for time-series processing, aimed at fault detection and classification in oil well operation. Building upon the experience with the original 3WToolkit system and leveraging the Petrobras <a href="https://github.com/petrobras/3W">3W Dataset</a>, 3WToolkit+ introduces enhanced functionalities, such as advanced data imputation, deep feature extraction, synthetic data augmentation, and high-performance computing capabilities for model training.</p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img width="400" src="https://coppe.ufrj.br/wp-content/uploads/2023/10/COPPE-horiz-completa-cor-300dpi.jpg">
    <img width="600" src="https://sites.ufpe.br/litpeg/wp-content/uploads/sites/10/2022/06/Petrobras-Logo.png">
</div>


<p style="text-align: justify;">The development of the 3WToolkit+ is the result of a collaborative partnership between Petrobras, with a focus on the CENPES research center, and the COPPE/Universidade Federal do Rio de Janeiro (UFRJ). This joint effort brings together complementary strengths: COPPE/UFRJ contributes decades of proven expertise in signal processing and machine learning model development, while CENPES offers access to highly specialized technical knowledge and real-world operational challenges in the oil and gas sector. This synergy ensures that 3WToolkit+ is both scientifically rigorous and practically relevant, addressing complex scenarios with robust and scalable AI-based solutions for time-series analysis and fault detection in oil well operations.</p>


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

## Documentation <a id="documentation"></a>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/3w_top_view.png">

<p style="text-align: justify;">The image above illustrates the high-level architecture of the 3WToolkit+, designed to support the full pipeline of machine learning applications using the 3W dataset—from raw data ingestion to model evaluation and delivery to end users. Each block in the architecture is briefly described below:</p>

### 3W Dataset Versions <a id="dataset_versions"></a>
<p style="text-align: justify;">This block represents different available versions of the 3W dataset, which include real and simulated data from offshore oil wells. These datasets serve as the foundation for all subsequent stages of data processing, modeling, and evaluation.</p>

### Data Loader <a id="data_loader"></a>
<p style="text-align: justify;">The Data Loader module is responsible for importing, validating, and preparing the raw 3W data for use in model training and evaluation. It handles missing data, standardizes variable formats, and performs initial quality checks to ensure compatibility across toolkit components.</p>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/3w_data_loader.png">

### Model Development <a id="model_dev"></a>
<p style="text-align: justify;">This central module provides the infrastructure for designing, training, and optimizing machine learning models for fault detection and classification. It supports both classical and deep learning models and includes tools for hyperparameter tuning, cross-validation, and model versioning.</p>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/3w_model_dev.png">

### Assessment <a id="model_assessment"></a>
<p style="text-align: justify;">The Assessment module evaluates model performance using both sample-level and event-level metrics. It includes support for traditional indicators (e.g., accuracy, precision, recall) as well as domain-specific metrics such as detection lag and anticipation time, which are critical for condition-based monitoring.</p>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/3w_assessment.png">

## Usage Documentation <a id="documentation2"></a>

### 3W Examples <a id="toolkit_examples"></a>
<p style="text-align: justify;">A curated set of ready-to-use model configurations and scripts that demonstrate how to apply the toolkit to common fault detection tasks using the 3W dataset. These examples accelerate onboarding and reproducibility.</p>

### 3W Tutorials/Demos <a id="toolkit_demos"></a>
<p style="text-align: justify;">Step-by-step tutorials and demonstration notebooks that guide users through the toolkit’s functionalities, explaining how each module operates and how to configure different experiments.</p>

### 3W Challenges <a id="toolkit_challenges"></a>
<p style="text-align: justify;">This component provides benchmarking tasks and open challenges using real scenarios derived from the 3W dataset. It promotes collaborative development and comparative evaluation of machine learning solutions in fault diagnosis.</p>

### 3W Videos <a id="toolkit_videos"></a>
<p style="text-align: justify;">Instructional videos that explain toolkit concepts, walk through complete modeling pipelines, and offer insights from domain experts. These videos aim to broaden accessibility and support training initiatives.</p>

## Toolkit UML <a id="uml"></a>

<p style="text-align: justify;">Building upon the high-level block diagram architecture, a detailed UML (Unified Modeling Language) diagram was developed to support the software engineering and implementation of the 3WToolkit+. The UML model formalizes the relationships between components, data structures, and workflows described in the block-level architecture, enabling a structured and maintainable development process.</p>

<p style="text-align: justify;">This transition from conceptual blocks to formal UML design ensures that each module—such as the Data Loader, Model Development, and Assessment—has clearly defined interfaces, class responsibilities, and interaction protocols. It also facilitates modular programming, unit testing, and future extensibility of the toolkit by providing developers with a shared, consistent blueprint for implementation.</p>

<p style="text-align: justify;">The UML diagram serves not only as an internal reference for the development team but also as part of the developer-oriented documentation that accompanies the toolkit and it is shown bellow</p>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/3w_toolkit_uml.png">

## Toolkit Setup  <a id="setup"></a>

### Docker <a id="docker"></a>

<p style="text-align: justify;">To ensure a consistent, reproducible, and isolated development environment, this project uses Docker as part of its core development workflow. Docker enables the encapsulation of all dependencies, configurations, and system-level requirements needed to run the application, eliminating the "it works on my machine" problem. By containerizing the development environment, we guarantee that all contributors and automated CI/CD pipelines operate under the same conditions, improving reliability and minimizing unexpected behaviors. Additionally, Docker simplifies environment setup, allowing developers to start contributing quickly without manually installing and configuring complex dependencies. This approach also facilitates testing across multiple versions of Python or system libraries when needed, supporting robust and portable software engineering practices.</p>

<img width="1200" src="https://github.com/Mathtzt/3WToolkit/blob/main/docs/figures/docker-logo-blue.png">

<p style="text-align: justify;">All dependencies and system requirements for this project have been fully encapsulated within a Docker image to ensure consistency and reproducibility across environments. As such, it is highly recommended that developers use this Docker image during development. You can either build the image locally or pull it directly from Docker Hub, depending on your preference or workflow.</p>

<p style="text-align: justify;">Docker operates by leveraging containerization, which allows applications and their dependencies to run in isolated user-space environments that share the host system's kernel. Unlike traditional virtual machines, which emulate entire hardware stacks and run full guest operating systems, Docker containers are significantly more lightweight and faster to start. This leads to improved resource efficiency, lower overhead, and greater scalability. In development environments where multiple users are working on the same codebase, Docker provides a critical advantage: it ensures that all contributors run the exact same environment, from system libraries to Python packages, without the need for heavy virtual machines or complex configuration. Containers can be spun up instantly, consume fewer resources, and integrate seamlessly with CI/CD pipelines. Moreover, Docker images can be versioned, shared via registries like Docker Hub, and easily rebuilt, enabling collaborative and reproducible workflows across diverse teams and systems.</p>


#### Build a docker image locally
<p style="text-align: justify;">To build the Docker image locally, navigate to the root directory of the project and run:</p>
```bash
docker build --tag=<usr name>/3w_tk_img:latest .
```

#### Push a docker image from DockerHub
<p style="text-align: justify;">To push the image to <a href="https://hub.docker.com/r/natmourajr/3w_tk_img">Docker Hub</a>, make sure you are logged in and then execute:</p>

```bash
docker pull natmourajr/3w_tk_img
```

#### Run a docker image locally
After building or pulling the image in computer, just run:

```bash
docker run  natmourajr/3w_tk_img
```

### Development in VSCode using Docker <a id="vscode_docker"></a>

1. VSCode extension: Dev Containers (ID: `ms-vscode-remote.remote-containers`).
2. Open your project root folder (`3WToolkit/`) in VSCode.
3. Press `F1` or `Ctrl+Shift+P` and select:

   ```
   Dev Containers: Open Folder in Container
   ```
4. VSCode will build the image and open your project *inside the container*.
5. Working Inside the Container:
    - Once the container is running, it is possible to use the **VSCode terminal**, which now runs inside the container.

Note:
Install libraries using `pip` will stay isolated from your host system.

### Requirements  <a id="requirements"></a>

<p style="text-align: justify;">This project uses <a href="https://python-poetry.org/">Poetry</a> as its dependency and packaging manager to ensure a consistent, reliable, and modern Python development workflow. Poetry simplifies the management of project dependencies by providing a single `pyproject.toml` file to declare packages, development tools, and metadata, while automatically resolving compatible versions. Unlike traditional `requirements.txt` workflows, Poetry creates an isolated and deterministic environment using a lock file (`poetry.lock`), ensuring that all contributors and deployment environments use exactly the same package versions. It also streamlines publishing to PyPI, virtual environment creation, and script execution, making it a comprehensive tool for managing the entire lifecycle of a Python project. By adopting Poetry, we reduce the risk of dependency conflicts and improve the reproducibility and maintainability of the codebase.</p>

### Installation <a id="install"></a>

#### Python
It is possible to perform the installation in three different ways.

1. **ThreeWToolkit** is on PyPI, so you can use pip to install it:

```
pip install ThreeWToolkit
```

2. **Installing directly from the git repository (private):**
You can install directly using:

```
pip install git+https://github.com/Mathtzt/3WToolkit.git
```

Note: *Authentication is required*.

3. **Installing via `requirements.txt`:**
You can install using:

```
pip install -r requirements.txt
```

Note: *Authentication is required*.

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