The list of future developments for the collaboration project between MAIS system (by SMT-UFRJ) and 3W Project (by Petrobras) is detailed below:

Short term:
* Implement the one-vs-all fault-detection mode (experiment type #1);
* Implement normal-vs-fault_i fault detection/classification mode (experiment type #2);
* Incorporate feature importance evaluation module, including classical methods (boruta, shap, mrmr, rfe);
* Incorporate output evaluation module, which includes custom metrics and graphical outputs;

Long term:
* Incorporate missing-data imputation module (still under development);
* Implement new modeling strategy using an ensemble of binary classifiers (one for each class) and combining their respective outputs;
* Incorporate and test other clssification algorithms: xgboost, tabnet;
* Use a temporal approach following a deep-learning-like strategy. In this approach, one feeds the system directly with the sensor signals, instead of extracting features before inputing them to the classifiers.