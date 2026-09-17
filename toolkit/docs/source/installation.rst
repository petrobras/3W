============
Installation
============

This guide covers the prerequisites and step-by-step instructions for installing **ThreeWToolkit**.

It is possible to perform the installation in different ways, depending on what you want to do:

- **Just want to use the ThreeWToolkit?** Install the published package from PyPI. You don't need to clone the repository.
- **Want to develop or contribute to the ThreeWToolkit?** Clone (or fork) the repository and install it locally in editable mode, so your changes are picked up immediately without reinstalling.

Prerequisites
=============

Ensure your environment meets the following baseline requirements:

* **Python**: Version 3.10 or higher.
* **Operating System**: Linux, macOS, or Windows (64-bit).
* **Package Manager**: ``pip`` or ``uv`` (recommended for fast dependency resolution).

Direct Installation via PyPI
============================

.. note::
   The package name used for installation differs slightly from the project name. Because PyPI and ``pip`` do not allow package names starting with a digit, use **ThreeWToolkit** for installation commands (e.g., ``pip install ThreeWToolkit``) and dependency specification, whereas **3W Toolkit** refers to the project name and its features.

The simplest and recommended way for end users to install the latest released version of **ThreeWToolkit** is directly via ``pip``:

.. include:: ../../ThreeWToolkit/README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start-installation-opt-a -->
   :end-before: <!-- end-installation-opt-a -->

Optional Dependencies (Extras)
==============================

**ThreeWToolkit** provides optional dependency sets for specific workflows (such as development, documentation building, image generation, or advanced ML backends).

You can install them by appending the extra group inside brackets (use quotes to prevent shell issues in environments like ``zsh``):

* **Development suite** (testing, linting, type-checking):

  .. code-block:: bash

     pip install "ThreeWToolkit[dev]"

* **Documentation tools** (Sphinx, MyST Parser, RTD theme):

  .. code-block:: bash

     pip install "ThreeWToolkit[docs]"

* **Image & SVG processing tools**:

  .. code-block:: bash

     pip install "ThreeWToolkit[images]"

* **Jupyter Notebook integration**:

  .. code-block:: bash

     pip install "ThreeWToolkit[notebooks]"

* **Computer Vision & Advanced Deep Learning extras** (Torchvision, Timm, Torchmetrics):

  .. code-block:: bash

     pip install "ThreeWToolkit[torch-extras]"

* **Image processing algorithms** (Scikit-Image):

  .. code-block:: bash

     pip install "ThreeWToolkit[scikit-extras]"

Combining Multiple Extras
-------------------------

To set up a full development environment with docs and image tools combined, pass a comma-separated list:

.. code-block:: bash

   pip install "ThreeWToolkit[dev,docs,images,notebooks]"

Development Installation (Source)
=================================

If you plan to contribute to the package, access the latest features on the development branch, or run experimental pipelines, install the toolkit from source in editable mode.

1. Fork or Clone the Repository
--------------------------------

Option A: Clone Official Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using SSH:

.. code-block:: bash

   git clone git@github.com:petrobras/3W.git
   cd 3W/toolkit/ThreeWToolkit

Using HTTPS:

.. code-block:: bash

   git clone https://github.com/petrobras/3W.git
   cd 3W/toolkit/ThreeWToolkit

Option B: Fork and Clone
~~~~~~~~~~~~~~~~~~~~~~~~

1. Go to `https://github.com/petrobras/3W <https://github.com/petrobras/3W>`_ and click **Fork**.
2. Clone your personal fork locally:

.. code-block:: bash

   git clone git@github.com:<your-username>/3W.git
   cd 3W/toolkit/ThreeWToolkit

2. Install in Editable Mode
---------------------------

.. include:: ../../ThreeWToolkit/README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start-installation-editable -->
   :end-before: <!-- end-installation-editable -->

Verifying the Installation
==========================

Verify that the installation was successful by checking the package version in Python:

.. code-block:: python

   import ThreeWToolkit

   print(ThreeWToolkit.__version__)