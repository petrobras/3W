# Modular Artificial Inteligence System (MAIS)

This repository presents MAIS, a system that implements Machine Learning techniques on a modular way, enabling the developer to test his/her own experiments and/or adapting others esperiments with their own idea. MAIS was developed by the Signal, Multimedia and Telecommunications (SMT) laboratory with the help from Petrobras.

In this version, MAIS implements a multiclass LGBM classifier, with the following optional features:

* Statistical features
  * Regular average withn an window;
  * Exponetially weigthed average within an window.
* Wavelets features
* Imputation methods: keep NaN values, change by the mean, ...
* Different labeling methods
  * Using the most recent label from an window as the lael for that sample; or
  * Using the label in the middle of an window as the lael for that sample.
* Feature selection using Random Forest importance

# Repository Structure

```
├── environment.yml
├── experiments
│   └── multiclass
│       ├── experiments
│       │   ├── base_experiment.py
│       │   ├── multi_ew_stats_mrl_nonan.py
│       │   └── ...
│       ├── train_lgbm.py
│       └── tune_lgbm.py
├── mais
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── feature_mappers.py
│   │   ├── label_mappers.py
│   │   └── utils.py
│   └── visualization
│       ├── __init__.py
│       └── generate_report.py
└── setup.py
```
MAIS uses a class called Experiment, which contains all the necessary steps to create your experiment. So, under the folder "experiments/multiclass/experiments/", you add your custom Experiment class, based on the BaseExperiment defined on "experiments/multiclass/experiments/base_experiment.py". Some examples are already implemented in order to give an idea on how an experiment is created.

The "mais/" folder contains classes definitions that create everything that is used to create an experiment, i.e., contains all utility classes . Some of the 

  1. mais/data/dataset.py: Defines the class MAEDataset, which contains the core logic behind MAIS dataloader. Some of its functions are: read a .csv, read the feature extraction, create the final table (the model input).
  2. mais/data/feature\_mappers.py: Defines the classes that extract the attributes for a given experiment. the implementation uses torch in order to make the extraction faster when using a lot of data. In the current version there are some strategies already implemented, for example: 
     1. TorchStatisticalFeatureMapper: created statistical features from a rectangular window;
     2. TorchWaveletFeatureMapper: creates wavelets features;
     3. TorchEWStatisticalFeatureMapper: creates statistical features from a window with exponential weights for each sample.
  3. mais/data/label\_mappers.py: Creates the classes that define how the detection is done. For example, it is possible to choose if the transient period of signal will be considered, or if the samples in the beggining of a file (which are usually not faulty) will be considered.

  So, in order to add new utility functions and/or classes, the "mais/" folder is probably the best place (under the correspondent file). For example, if one needs to create a new feature extractor, the best way to proceed is creating a new FeatureMapper under the file "mais/data/feature\_mappers.py".

# Experiment examples

In the folder experiments/multiclass/ there are many examples that can guide on how to create a new one:
1. multi_ew_stats_mrl_nonan.py
2. multi_mixed_mrl_nonan.py
3. multi_mixed_select_mrl_nonan.py
4. multi_stats_mrl_nonan.py
5. multi_stats_select_mrl_nonan.py
6. multi_wavelets_mrl_nonan.py
7. multi_wavelets_select_mrl_nonan.py

The name of theses experiments reflect what they implements, for examples, the experiment "multi_stats_select_mrl_nonan.py" implements a multiclass classifier that uses statistical features, a feature selector, uses the most recent label as the label associated to a window and imputes NaN values. The acronym we used are:

* multi = Multiclass Experiment;
* ew = Exponentially weighted;
* stats = Statistical features;
* mrl = Most recent label;
* nonan = NaN imputation;
* mixed = both statistical and wavelet features; 
* select = Feature selector; and
* wavelets = Wavelets
# How to use

After creating the experiment and putting it into the experiment folder (for example, 'experiments/multiclass/experiments/example.py', 

  1. [OPTIONAL] Initialize a mlflow server with an sqlite URI for the logs. In general, this option is the best one to avoid mlflow from creating tons of files.
  2. Execute 'tune\_lgbm.py'. This script initialize a mlflow experiment containing all runs from the Bayesian optimization search. For every run, the script trains a model and saves its metrics. Its commands are:
     1. data-root: Root directory with the data;
     2. experiment-name: Name of the experiment (must be inside 'experiments/multiclass/experiments/');
     3. num-trials: Number of Bayesian optimization trials;
     4. inner-splits: Number of cross-validation inner loops;
     5. outer-splits: Number of cross-validation outer loops;
     6. n-jobs: Number of cores available for parallel processing;
  All this commands can be also consulted using --help. [P.S: Use apropriate environment variables for your mlflow log system.]