==========
Evaluation
==========

**ThreeWToolkit** provides a dedicated evaluation workflow for assessing
trained models on unseen data. The workflow separates model training,
prediction, metric calculation, and result visualization.

The general evaluation workflow is:

.. code-block:: text

   Training Data
        |
        v
   Model Training
        |
        v
   Training Results
        |
        +-------------------+
        |                   |
        v                   v
   Validation Data      Test Data
                            |
                            v
                       Predictions
                            |
                            v
                       Assessment
                            |
                            v
                     Metrics & Plots


Training Results
================

After preparing the training and validation datasets, a trainer can be
used to train the model:

.. code-block:: python

   train_results = trainer.train(
       train_dataset=ds_train_transformed,
       val_dataset=ds_val_transformed,
   )

The resulting ``train_results`` object contains the training history,
including the training and validation losses. These values can be used
to analyze the model's learning behavior across epochs.

For example:

.. code-block:: python

   history = train_results.history

   plt.figure(figsize=(10, 5))
   plt.plot(history.val_loss, label="Val Loss")
   plt.plot(history.train_loss, label="Train Loss")
   plt.title("Training and Validation Loss")
   plt.xlabel("Epoch")
   plt.ylabel("Loss")
   plt.legend()


Test Predictions
================

Once the model has been trained, predictions on unseen test data can be
obtained using the same trainer instance:

.. code-block:: python

   test_results = trainer.predict(ds_test_transformed)

The resulting object contains the model predictions and the
corresponding target values, which can then be passed to the assessment
pipeline.


Model Assessment
================

Model performance can be evaluated using ``ModelAssessmentConfig``.
The assessment configuration defines the metrics to be calculated and
provides an evaluation interface for the model predictions.

For example:

.. code-block:: python

   assessment = ModelAssessmentConfig(
       metrics=["accuracy"],
   ).build()

   results = assessment.evaluate(
       training_results=train_results,
       predictions=test_results,
   )

The ``training_results`` argument provides information from the training
process, while ``predictions`` contains the results obtained on the
test set.

Multiple metrics can be specified through the ``metrics`` configuration
according to the requirements of the evaluation task.


Assessment Visualization
========================

Evaluation results can also be visualized using
``AssessmentVisualizationConfig``. The visualization component provides
methods for inspecting model predictions and assessment results.

For classification tasks, a confusion matrix can be generated from the
true and predicted labels:

.. code-block:: python

   plotter = AssessmentVisualizationConfig().build()

   fig = plotter.plot_confusion_matrix(
       y_true=test_results.y_true,
       y_pred=test_results.y_pred,
       normalize=True,
       title="Normalized Confusion Matrix",
   )

The ``normalize`` option can be used to display the confusion matrix
using normalized values, making it easier to compare the distribution
of predictions across classes.

Evaluation Workflow
===================

A complete evaluation workflow therefore consists of four main stages:

1. Train the model using training and validation data.
2. Generate predictions for the test dataset.
3. Evaluate the predictions using ``ModelAssessmentConfig``.
4. Visualize the resulting predictions using
   ``AssessmentVisualizationConfig``.

This separation allows the same training and prediction results to be
used by different assessment and visualization components.

For a complete example of model assessment, including the preparation
of the datasets and interpretation of the results, see the
:doc:`../_demos_gen/11_assessment_examples` tutorial.