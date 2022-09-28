# Binary classifier of Spurious Closure of DHSV

The main aspects of this problem are:

* It is a binary classifier in the sense that labels associated with Spurious Closure of DHSV are considered examples of the positive class and all other labels are considered examples of the negative class;
* It is an OVA (one versus all) classifier. The negative class has examples extracted from normal and all other event types present in the 3W dataset;
* Spurious Closure of DHSV transients are treated as different from Spurious Closure of DHSV steady state;
* Only real instances are used.

To submit a **pull request** with the approach you developed for this specific problem, please follow the specifications below.

1. Your [Jupyter Notebooks](https://jupyter.org/) must be named as `problems\01_binary_classifier_of_spurious_closure_of_dhsv\[your_name_here]\main.ipynb`;
1. You can include any module or feature you need in this subdirectory;
1. If necessary, include in this subdirectory a `environment.yml` generated with [conda](https://docs.conda.io) with all used packages and their versions;
1. The `.ipynb_checkpoints` must not be included in your **pull request**;
1. For each test sample you need to estimate an array of probabilities associated with the following labels:
    * Negative class;
    * Transient of the positive class;
    * Steady state of the positive class.
1. You can add figures and text to explain your approach.