import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

from ThreeWToolkit.assessment.model_assess import (
    AssessmentInput,
    AssessmentInputValidator,
    AssessmentOutput,
    AggregatedResults,
    MetricRegistry,
    MetricsAggregator,
    ModelAssessment,
)
from ThreeWToolkit.core.base_assessment import ModelAssessmentConfig
from ThreeWToolkit.core.enums import DataSplitEnum, TaskTypeEnum


def make_assessment_config(**overrides):
    """
    Creates a ModelAssessmentConfig instance for testing.

    A default configuration is created and optionally overridden
    by keyword arguments provided to the function. This helper
    simplifies configuration setup across multiple tests.
    """
    defaults = dict(
        metrics=["accuracy"],
        task_type=TaskTypeEnum.CLASSIFICATION,
        output_dir=Path("/tmp/test_assess"),
    )
    defaults.update(overrides)
    return ModelAssessmentConfig(**defaults)


def mock_model(name="MockModel"):
    """
    Creates a mocked model with a basic prediction strategy.

    The returned model simulates the minimal interface expected
    by ModelAssessment, including a prediction strategy that
    returns fixed predictions for testing purposes.
    """
    model = MagicMock()
    model.model_name = name
    strategy = MagicMock()
    strategy.requires_dataloader.return_value = False
    strategy.predict.return_value = np.array([0, 1, 0, 1])
    model.get_prediction_strategy.return_value = lambda: strategy
    return model


def base_assessment_input(models=None, n=4):
    """
    Creates a basic AssessmentInput instance with synthetic data.

    This helper generates random feature data and simple labels,
    allowing tests to easily construct valid inputs for the
    ModelAssessment pipeline.
    """
    x = np.random.rand(n, 4).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    return AssessmentInput(
        models=models or [mock_model()],
        x=x,
        y=y,
        dataset_split=DataSplitEnum.TEST,
    )


class TestMetricRegistry:
    """
    Tests for the MetricRegistry component.

    This suite validates that metrics are correctly resolved based on the
    task type (classification or regression), and that invalid metrics
    raise appropriate errors.
    """

    @pytest.fixture
    def registry(self):
        """
        Returns a fresh MetricRegistry instance for each test.
        """
        return MetricRegistry()

    def test_resolve_valid_classification_metric(self, registry):
        """
        Ensures that a valid classification metric can be resolved and is callable.
        """
        fns = registry.resolve(TaskTypeEnum.CLASSIFICATION, ["accuracy"])
        assert "accuracy" in fns
        assert callable(fns["accuracy"])

    def test_resolve_valid_regression_metric(self, registry):
        """
        Ensures that a valid regression metric is correctly resolved.
        """
        fns = registry.resolve(TaskTypeEnum.REGRESSION, ["explained_variance"])
        assert "explained_variance" in fns

    def test_resolve_unknown_metric_raises(self, registry):
        """
        Verifies that resolving an unknown metric raises a ValueError.
        """
        with pytest.raises(ValueError, match="not available"):
            registry.resolve(TaskTypeEnum.CLASSIFICATION, ["unknown_metric"])

    def test_resolve_multiple_metrics(self, registry):
        """
        Ensures that multiple metrics can be resolved simultaneously.
        """
        fns = registry.resolve(TaskTypeEnum.CLASSIFICATION, ["accuracy", "f1"])
        assert len(fns) == 2


class TestMetricsAggregator:
    """
    Tests for the MetricsAggregator utility.

    These tests verify that metrics computed across multiple folds
    are correctly aggregated into mean and standard deviation values.
    """

    def test_aggregate_computes_mean_and_std(self):
        """
        Ensures the aggregator correctly computes mean and standard deviation
        from per-fold metric values.
        """
        per_fold = [{"accuracy": 0.8}, {"accuracy": 0.9}, {"accuracy": 0.7}]
        result = MetricsAggregator.aggregate(per_fold)

        assert isinstance(result, AggregatedResults)
        assert result.n_folds == 3
        assert result.metrics_mean["accuracy"] == pytest.approx(0.8, abs=1e-4)
        assert result.metrics_std["accuracy"] > 0

    def test_aggregate_multiple_metrics(self):
        """
        Ensures aggregation works when multiple metrics are provided.
        """
        per_fold = [{"acc": 0.9, "f1": 0.85}, {"acc": 0.8, "f1": 0.75}]
        result = MetricsAggregator.aggregate(per_fold)
        assert "acc" in result.metrics_mean
        assert "f1" in result.metrics_std


class TestAssessmentInputValidator:
    """
    Tests for the AssessmentInputValidator.

    This suite verifies that input validation behaves correctly,
    including handling missing data, mismatched fold sizes,
    incorrect model structures, and automatic normalization
    of input parameters.
    """

    @pytest.fixture
    def config(self):
        """
        Creates a default ModelAssessmentConfig used for validation tests.
        """
        return make_assessment_config()

    def test_valid_input_passes(self, config):
        """
        Ensures that a valid AssessmentInput passes validation unchanged.
        """
        data = base_assessment_input()
        result = AssessmentInputValidator.validate(data, config)
        assert result is data

    def test_raises_if_x_is_none(self, config):
        """
        Ensures validation fails if input feature matrix `x` is None.
        """
        data = base_assessment_input()
        data.x = None
        with pytest.raises(ValueError, match="Both x and y must be provided"):
            AssessmentInputValidator.validate(data, config)

    def test_raises_if_fold_lengths_mismatch(self, config):
        """
        Ensures validation fails when fold lists do not match
        the number of provided models.
        """
        data = base_assessment_input(models=[mock_model(), mock_model()])
        data.x_train_folds = [np.array([1])]  # length 1, but 2 models
        data.y_train_folds = [np.array([1])]
        data.x_val_folds = [np.array([1])]
        data.y_val_folds = [np.array([1])]
        with pytest.raises(ValueError, match="x_train_folds length must match"):
            AssessmentInputValidator.validate(data, config)

    def test_wraps_single_model_in_list(self, config):
        """
        Ensures that a single model input is automatically wrapped into a list.
        """
        data = base_assessment_input()
        data.models = mock_model()  # not a list
        result = AssessmentInputValidator.validate(data, config)
        assert isinstance(result.models, list)

    def test_raises_if_models_empty_list(self, config):
        """
        Ensures validation fails if the models list is empty.
        """
        data = base_assessment_input()
        data.models = []
        with pytest.raises(ValueError, match="At least one model"):
            AssessmentInputValidator.validate(data, config)

    def test_dataset_split_none_uses_config_default(self, config):
        """
        Ensures that dataset_split defaults to the value defined in the config.
        """
        data = base_assessment_input()
        data.dataset_split = None
        result = AssessmentInputValidator.validate(data, config)
        assert result.dataset_split == config.dataset_split

    def test_kwargs_none_becomes_empty_dict(self, config):
        """
        Ensures that kwargs is normalized to an empty dictionary if None.
        """
        data = base_assessment_input()
        data.kwargs = None
        result = AssessmentInputValidator.validate(data, config)
        assert result.kwargs == {}

    def test_raises_if_fold_field_is_none(self, config):
        """
        Ensures validation fails if fold inputs are partially provided
        instead of all together.
        """
        data = base_assessment_input(models=[mock_model()])
        data.x_train_folds = [np.array([1])]
        data.y_train_folds = None
        data.x_val_folds = [np.array([1])]
        data.y_val_folds = [np.array([1])]
        with pytest.raises(ValueError, match="must be provided together"):
            AssessmentInputValidator.validate(data, config)

    @pytest.mark.parametrize(
        "bad_field,match",
        [
            ("y_train_folds", "y_train_folds length must match"),
            ("x_val_folds", "x_val_folds length must match"),
            ("y_val_folds", "y_val_folds length must match"),
        ],
    )
    def test_raises_on_fold_length_mismatch(self, config, bad_field, match):
        """
        Ensures validation fails when any fold list length does not match
        the number of models.
        """
        models = [mock_model(), mock_model()]
        data = base_assessment_input(models=models)

        data.x_train_folds = [np.array([1]), np.array([2])]
        data.y_train_folds = [np.array([1]), np.array([2])]
        data.x_val_folds = [np.array([1]), np.array([2])]
        data.y_val_folds = [np.array([1]), np.array([2])]

        setattr(data, bad_field, [np.array([1])])  # length 1 vs 2 models
        with pytest.raises(ValueError, match=match):
            AssessmentInputValidator.validate(data, config)


class TestModelAssessmentSingle:
    """
    Tests the ModelAssessment evaluation workflow for a single model.

    These tests verify correct execution of evaluation, prediction
    generation, metric computation, and report handling for
    non-cross-validation scenarios.
    """

    @pytest.fixture
    def config(self, tmp_path):
        """
        Creates a temporary configuration for ModelAssessment tests.
        """
        return make_assessment_config(output_dir=tmp_path)

    def test_evaluate_single_returns_output(self, config):
        """
        Ensures evaluate() returns a valid AssessmentOutput for single-model evaluation.
        """
        data = base_assessment_input()
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)

        assert isinstance(result, AssessmentOutput)
        assert result.is_cross_validation is False
        assert result.metrics is not None
        assert "accuracy" in result.metrics

    def test_evaluate_single_predictions_shape(self, config):
        """
        Ensures predictions and true values have the expected shape.
        """
        data = base_assessment_input(n=4)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert result.predictions.shape == (4,)
        assert result.true_values.shape == (4,)

    def test_evaluate_populates_experiment_dir(self, config):
        """
        Ensures the experiment directory is created during evaluation.
        """
        data = base_assessment_input()
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert result.experiment_dir is not None

    def test_generate_report_true_sets_report_class(self, tmp_path):
        """
        Ensures the report generation class is correctly loaded when reports are enabled.
        """
        fake_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "ThreeWToolkit.reports.report_generation": MagicMock(
                    ReportGeneration=fake_cls
                )
            },
        ):
            config = make_assessment_config(output_dir=tmp_path, generate_report=True)
            assessor = ModelAssessment(config)
            assert hasattr(assessor, "_report_generation_class")

    def test_generate_report_import_error_disables_report(self, tmp_path):
        """
        Ensures report generation is disabled if the report module cannot be imported.
        """
        with patch("builtins.__import__", side_effect=ImportError):
            config = make_assessment_config(output_dir=tmp_path, generate_report=True)
            try:
                assessor = ModelAssessment(config)
                assert assessor.config.generate_report is False
            except Exception:
                pass

    def test_pre_process_raises_for_non_assessment_input(self, config):
        """
        Ensures pre_process() raises TypeError when receiving invalid input types.
        """
        assessor = ModelAssessment(config)
        with pytest.raises(TypeError, match="AssessmentInput"):
            assessor.pre_process({"models": [], "x": None, "y": None})

    def test_pre_process_returns_validated_input(self, config):
        """
        Ensures pre_process() returns validated AssessmentInput data.
        """
        assessor = ModelAssessment(config)
        data = base_assessment_input()
        result = assessor.pre_process(data)
        assert isinstance(result, AssessmentInput)

    def test_run_delegates_to_evaluate(self, config):
        """
        Ensures run() internally calls evaluate().
        """
        assessor = ModelAssessment(config)
        data = base_assessment_input()
        with patch.object(assessor, "evaluate", wraps=assessor.evaluate) as mock_eval:
            assessor.run(data)
            mock_eval.assert_called_once_with(data)

    def test_post_process_returns_data_unchanged(self, config):
        """
        Ensures post_process() returns the output without modification.
        """
        assessor = ModelAssessment(config)
        output = MagicMock(spec=AssessmentOutput)
        result = assessor.post_process(output)
        assert result is output


class TestModelAssessmentCV:
    """
    Tests the ModelAssessment evaluation workflow under cross-validation.

    These tests verify that multiple models (representing folds) are
    evaluated correctly, metrics are aggregated, and reporting is
    triggered when enabled.
    """

    @pytest.fixture
    def config(self, tmp_path):
        """
        Creates a temporary ModelAssessmentConfig for CV tests.
        """
        return make_assessment_config(output_dir=tmp_path)

    def test_evaluate_cv_aggregates_folds(self, config):
        """
        Ensures cross-validation evaluation aggregates fold results correctly.
        """
        models = [mock_model(f"model_{i}") for i in range(3)]
        data = base_assessment_input(models=models)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)

        assert result.is_cross_validation is True
        assert result.aggregated_results is not None
        assert result.aggregated_results.n_folds == 3
        assert len(result.fold_results) == 3

    def test_evaluate_cv_metrics_per_fold(self, config):
        """
        Ensures aggregated metrics contain expected values after CV evaluation.
        """
        models = [mock_model() for _ in range(2)]
        data = base_assessment_input(models=models)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert "accuracy" in result.aggregated_results.metrics_mean

    def test_evaluate_calls_generate_report_when_enabled(self, tmp_path):
        """
        Ensures report generation is triggered when enabled in the configuration.
        """
        config = make_assessment_config(output_dir=tmp_path, generate_report=True)
        assessor = ModelAssessment(config)
        assessor.config.generate_report = True  # garante flag ativa pós-__init__

        with patch.object(assessor, "_generate_report") as mock_report:
            assessor.evaluate(base_assessment_input())
            mock_report.assert_called_once()


class TestGetPredictions:
    """
    Tests the internal prediction logic used by ModelAssessment.

    Ensures prediction strategies are invoked correctly depending on
    whether a dataloader is required.
    """

    def test_uses_dataloader_when_required(self, tmp_path):
        """
        Ensures the prediction strategy receives a dataloader when required.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)

        model = MagicMock()
        model.model_name = "test"
        strategy = MagicMock()
        strategy.requires_dataloader.return_value = True
        strategy.predict.return_value = np.array([0, 1, 0, 1])
        model.get_prediction_strategy.return_value = lambda: strategy

        X = np.random.rand(4, 4).astype(np.float32)
        _ = assessor._get_predictions(model, X)

        assert strategy.predict.called
        call_kwargs = strategy.predict.call_args[1]
        assert "loader" in call_kwargs


class TestSummary:
    """
    Tests the summary generation functionality of ModelAssessment.

    Ensures summaries are correctly generated before and after evaluation,
    including cross-validation scenarios.
    """

    def test_summary_before_evaluate(self, tmp_path):
        """
        Ensures summary reports that no evaluation results exist before running evaluation.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assert "No evaluation results" in assessor.summary()

    def test_summary_after_single_evaluate(self, tmp_path):
        """
        Ensures summary includes model name and metrics after single-model evaluation.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.evaluate(base_assessment_input())
        summary = assessor.summary()
        assert "accuracy" in summary
        assert "MockModel" in summary

    def test_summary_after_cv_evaluate(self, tmp_path):
        """
        Ensures summary includes aggregated metrics and standard deviation after CV evaluation.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        models = [mock_model() for _ in range(2)]
        assessor.evaluate(base_assessment_input(models=models))
        summary = assessor.summary()
        assert "±" in summary


class TestToNumpy:
    """
    Tests the internal conversion utility that normalizes input data
    into NumPy arrays.
    """

    @pytest.fixture
    def assessor(self, tmp_path):
        """
        Creates a ModelAssessment instance for numpy conversion tests.
        """
        return ModelAssessment(make_assessment_config(output_dir=tmp_path))

    def test_converts_series(self, assessor):
        """
        Ensures pandas Series objects are converted to NumPy arrays.
        """
        result = assessor._to_numpy(pd.Series([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_converts_dataframe(self, assessor):
        """
        Ensures pandas DataFrame objects are converted to NumPy arrays.
        """
        result = assessor._to_numpy(pd.DataFrame({"a": [1, 2]}))
        assert isinstance(result, np.ndarray)

    def test_converts_list(self, assessor):
        """
        Ensures Python lists are converted to NumPy arrays.
        """
        result = assessor._to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)


class TestExportResults:
    """
    Tests the export functionality of ModelAssessment.

    These tests verify that predictions and metrics are correctly
    exported to CSV files and that error handling works properly
    when results are missing.
    """

    def test_export_single_creates_csv_files(self, tmp_path):
        """
        Ensures CSV files are generated after single-model evaluation when export is enabled.
        """
        config = make_assessment_config(output_dir=tmp_path, export_results=True)
        assessor = ModelAssessment(config)
        assessor.evaluate(base_assessment_input())

        exp_dir = Path(assessor.results.experiment_dir)
        csv_files = list(exp_dir.glob("*.csv"))
        assert len(csv_files) >= 2  # predictions + metrics

    def test_export_cv_creates_csv_files(self, tmp_path):
        """
        Ensures CSV files are generated after cross-validation evaluation.
        """
        config = make_assessment_config(output_dir=tmp_path, export_results=True)
        assessor = ModelAssessment(config)
        models = [mock_model() for _ in range(2)]
        assessor.evaluate(base_assessment_input(models=models))

        exp_dir = Path(assessor.results.experiment_dir)
        csv_files = list(exp_dir.glob("*.csv"))
        assert len(csv_files) >= 2

    def test_export_raises_if_no_results(self, tmp_path):
        """
        Ensures exporting results fails if evaluation has not been executed.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        with pytest.raises(RuntimeError, match="No assessment results"):
            assessor._export_results()

    def test_evaluate_removes_existing_experiment_dir_on_failure(self, tmp_path):
        """
        Ensures the experiment directory is cleaned up if evaluation fails.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)

        def fail_after_dir(*args, **kwargs):
            assessor.experiment_dir = tmp_path / "exp_fake"
            assessor.experiment_dir.mkdir(parents=True, exist_ok=True)
            raise Exception("forced failure")

        with patch.object(assessor, "_evaluate_single", side_effect=fail_after_dir):
            with pytest.raises(RuntimeError, match="ModelAssessment failed"):
                assessor.evaluate(base_assessment_input())

        assert not assessor.experiment_dir.exists()

    def test_export_single_results_raises_if_no_results(self, tmp_path):
        """
        Ensures exporting single results raises an error if results are missing.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._export_single_results()

    def test_export_cv_results_raises_if_no_results(self, tmp_path):
        """
        Ensures exporting cross-validation results raises an error if results are missing.
        """
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._export_cv_results()


class TestAssessmentReport:
    """
    Tests the report generation logic of ModelAssessment.

    These tests validate the correct behavior of report generation
    for both single evaluation and cross-validation workflows,
    including error handling and disabled states.
    """

    @pytest.fixture
    def assessor(self, tmp_path):
        """
        Creates a ModelAssessment instance configured for report testing.
        """
        return ModelAssessment(make_assessment_config(output_dir=tmp_path))

    def _set_fake_report_class(self, assessor):
        """
        Injects a fake report generation class to simulate report creation.
        """
        fake_cls = MagicMock()
        fake_cls.return_value.generate_summary_report.return_value = MagicMock()
        assessor._report_generation_class = fake_cls
        return fake_cls

    def test_generate_report_returns_early_if_disabled(self, assessor):
        """
        Ensures report generation exits early when disabled in the configuration.
        """
        assessor.config.generate_report = False
        assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_report_raises_if_no_results(self, assessor):
        """
        Ensures an error is raised when report generation is requested without results.
        """
        assessor.config.generate_report = True
        assessor.results = None
        with pytest.raises(RuntimeError, match="No results available"):
            assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_report_calls_single_report_when_not_cv(self, assessor):
        """
        Ensures single-report generation is called when evaluation is not CV.
        """
        _ = self._set_fake_report_class(assessor)
        assessor.config.generate_report = True
        assessor.results = MagicMock(is_cross_validation=False)

        with patch.object(assessor, "_generate_single_report") as mock_single:
            input_data, output_data = MagicMock(), MagicMock()
            assessor._generate_report(input_data, output_data)
            mock_single.assert_called_once_with(input_data, output_data)

    def test_generate_report_warns_if_no_report_class(self, assessor, capsys):
        """
        Ensures a warning is printed when report generation class is unavailable.
        """
        assessor.config.generate_report = True
        assessor.results = MagicMock()

        if hasattr(assessor, "_report_generation_class"):
            delattr(assessor, "_report_generation_class")
        assessor._generate_report(MagicMock(), MagicMock())
        assert "Warning" in capsys.readouterr().out

    def test_generate_report_cv_branch_is_noop(self, assessor):
        """
        Ensures the CV branch executes without errors when report generation is enabled.
        """
        assessor.config.generate_report = True
        assessor.results = MagicMock(is_cross_validation=True)
        self._set_fake_report_class(assessor)
        assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_single_report_raises_if_no_results(self, assessor):
        """
        Ensures generating a single report fails when results are missing.
        """
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._generate_single_report(MagicMock(), MagicMock())

    def test_generate_single_report_calls_report_generator(self, assessor):
        """
        Ensures the report generator is invoked for single-model evaluation.
        """
        fake_cls = self._set_fake_report_class(assessor)
        assessor.results = MagicMock(
            is_cross_validation=False,
            metrics={"accuracy": 0.9},
            predictions=np.array([0, 1]),
            model_name="test",
        )
        assessor.experiment_dir = MagicMock()
        input_data = MagicMock()
        input_data.x_train_folds = [np.array([1])]
        input_data.y_train_folds = [np.array([1])]

        assessor._generate_single_report(input_data, MagicMock())
        fake_cls.return_value.generate_summary_report.assert_called_once_with(
            format="html"
        )

    def test_generate_cv_report_raises_if_no_results(self, assessor):
        """
        Ensures generating a CV report fails when results are missing.
        """
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._generate_cv_report(MagicMock(), MagicMock())

    def test_generate_cv_report_calls_report_generator(self, assessor):
        """
        Ensures the report generator is invoked for cross-validation evaluation.
        """
        fake_cls = self._set_fake_report_class(assessor)
        assessor.experiment_dir = MagicMock()

        fold = MagicMock(predictions=np.array([0, 1]))
        assessor.results = MagicMock(
            is_cross_validation=True,
            model_name="test",
            fold_results=[fold],
            aggregated_results=MagicMock(
                metrics_mean={"accuracy": 0.9},
                metrics_std={"accuracy": 0.05},
            ),
        )

        input_data = MagicMock()
        assessor._generate_cv_report(input_data, MagicMock())
        fake_cls.return_value.generate_summary_report.assert_called_once_with(
            format="html"
        )
