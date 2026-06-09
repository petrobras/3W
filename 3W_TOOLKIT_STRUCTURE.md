The 3W Toolkit is a software package written in Python 3 and located in
the [toolkit](toolkit) directory.

The importable package is [toolkit/ThreeWToolkit](toolkit/ThreeWToolkit)
and is structured in the following submodules:

* **assessment**: model assessment and assessment visualization resources;
* **clustering**: clustering algorithms, distances, normalization,
  resampling, consensus, and quality helpers;
* **core**: base classes, shared interfaces, enums, and configuration-driven
  instantiation utilities used by the other submodules;
* **data_visualization**: plotting and visualization tools for 3W time
  series and derived analyses;
* **dataset**: dataset loaders, subsets, transformations, and dataset
  output abstractions;
* **feature_extraction**: feature extraction methods and adapters;
* **metrics**: classification and regression metrics;
* **models**: model wrappers and model implementations;
* **preprocessing**: data cleaning, normalization, remapping, imputation,
  label filling, and related preprocessing steps;
* **reports**: report-generation utilities and LaTeX assets;
* **trainer**: framework-specific training orchestration;
* **utils**: general utilities used across the toolkit.

Toolkit usage examples and demonstration notebooks are located in
[toolkit/demos](toolkit/demos). Contributions that add or change toolkit
code, tests, demos, or development workflow should also follow the
[3W Toolkit contributing guide](3W_TOOLKIT_CONTRIBUTING.md).
