.. ThreeWToolkit documentation master file, created by
   sphinx-quickstart on Tue Aug 11 15:48:34 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ThreeWToolkit documentation
===========================

.. image:: https://raw.githubusercontent.com/petrobras/3W/main/images/3w_logo.png
   :width: 80px
   :align: right
   :alt: ThreeWToolkit Logo

**ThreeWToolkit** is an open-source Python framework for time-series processing, early fault detection, and event classification in oil well and pipeline operations.

Developed as a standardized platform built on top of Petrobras's benchmark **3W Dataset**, the toolkit bridges raw physical sensor telemetry with modern machine learning paradigms. It enables researchers, data scientists, and engineers to run fully reproducible experiments and develop operational models.

Why ThreeWToolkit?
==================

* **Domain-Aware Pipelines**: Designed specifically for the heterogeneous composition and multi-source telemetry (real, simulated, and hand-drawn synthetic instances) of oil well operations.
* **Standardized Benchmarking**: Offers a unified framework to ensure fair, reproducible performance evaluations across the global **3W Community**.
* **Modern High-Performance Architecture**: Optimized for memory-efficient handling and fast querying of large-scale time-series data using native `.parquet` ingestion.

Project & Community
===================

The **ThreeWToolkit** is part of the **3W Project**, an initiative developed in partnership by:

* **Petrobras**
* **Signal, Multimedia and Telecommunications Laboratory (SMT)** – Federal University of Rio de Janeiro (UFRJ)
* **Signal Processing Laboratory (LPS)** – Federal University of Rio de Janeiro (UFRJ)

It actively supports the **3W Community**, an international collaboration of researchers, industry partners, and independent data scientists working on AI/ML solutions for energy systems.

Next Steps
==========

* **New to ThreeWToolkit?** Start with our :doc:`installation` and :doc:`quickstart` guides.
* **Exploring the framework?** Read the :doc:`user_guide/index` or dive into the :doc:`api`.
* **Want to contribute?** Check out the :doc:`developer_guide/index`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer_guide/index

Quick Links
===========

* :ref:`genindex`
* :ref:`modindex`
* :doc:`installation`
* :doc:`quickstart`
