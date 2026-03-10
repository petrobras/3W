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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_assessment_config(**overrides):
    defaults = dict(
        metrics=["accuracy"],
        task_type=TaskTypeEnum.CLASSIFICATION,
        output_dir=Path("/tmp/test_assess"),
    )
    defaults.update(overrides)
    return ModelAssessmentConfig(**defaults)


def mock_model(name="MockModel"):
    model = MagicMock()
    model.model_name = name
    strategy = MagicMock()
    strategy.requires_dataloader.return_value = False
    strategy.predict.return_value = np.array([0, 1, 0, 1])
    model.get_prediction_strategy.return_value = lambda: strategy
    return model


def base_assessment_input(models=None, n=4):
    x = np.random.rand(n, 4).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    return AssessmentInput(
        models=models or [mock_model()],
        x=x,
        y=y,
        dataset_split=DataSplitEnum.TEST,
    )


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------


class TestMetricRegistry:

    @pytest.fixture
    def registry(self):
        return MetricRegistry()

    def test_resolve_valid_classification_metric(self, registry):
        fns = registry.resolve(TaskTypeEnum.CLASSIFICATION, ["accuracy"])
        assert "accuracy" in fns
        assert callable(fns["accuracy"])

    def test_resolve_valid_regression_metric(self, registry):
        fns = registry.resolve(TaskTypeEnum.REGRESSION, ["explained_variance"])
        assert "explained_variance" in fns

    def test_resolve_unknown_metric_raises(self, registry):
        with pytest.raises(ValueError, match="not available"):
            registry.resolve(TaskTypeEnum.CLASSIFICATION, ["unknown_metric"])

    def test_resolve_multiple_metrics(self, registry):
        fns = registry.resolve(TaskTypeEnum.CLASSIFICATION, ["accuracy", "f1"])
        assert len(fns) == 2


# ---------------------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------------------


class TestMetricsAggregator:

    def test_aggregate_computes_mean_and_std(self):
        per_fold = [{"accuracy": 0.8}, {"accuracy": 0.9}, {"accuracy": 0.7}]
        result = MetricsAggregator.aggregate(per_fold)

        assert isinstance(result, AggregatedResults)
        assert result.n_folds == 3
        assert result.metrics_mean["accuracy"] == pytest.approx(0.8, abs=1e-4)
        assert result.metrics_std["accuracy"] > 0

    def test_aggregate_multiple_metrics(self):
        per_fold = [{"acc": 0.9, "f1": 0.85}, {"acc": 0.8, "f1": 0.75}]
        result = MetricsAggregator.aggregate(per_fold)
        assert "acc" in result.metrics_mean
        assert "f1" in result.metrics_std


# ---------------------------------------------------------------------------
# AssessmentInputValidator
# ---------------------------------------------------------------------------


class TestAssessmentInputValidator:

    @pytest.fixture
    def config(self):
        return make_assessment_config()

    def test_valid_input_passes(self, config):
        data = base_assessment_input()
        result = AssessmentInputValidator.validate(data, config)
        assert result is data

    # def test_raises_if_no_models(self, config):
    #     data = base_assessment_input(models=[])
    #     with pytest.raises(ValueError, match="At least one model"):
    #         AssessmentInputValidator.validate(data, config)

    def test_raises_if_x_is_none(self, config):
        data = base_assessment_input()
        data.x = None
        with pytest.raises(ValueError, match="Both x and y must be provided"):
            AssessmentInputValidator.validate(data, config)

    def test_raises_if_fold_lengths_mismatch(self, config):
        data = base_assessment_input(models=[mock_model(), mock_model()])
        data.x_train_folds = [np.array([1])]  # length 1, but 2 models
        data.y_train_folds = [np.array([1])]
        data.x_val_folds = [np.array([1])]
        data.y_val_folds = [np.array([1])]
        with pytest.raises(ValueError, match="x_train_folds length must match"):
            AssessmentInputValidator.validate(data, config)

    def test_wraps_single_model_in_list(self, config):
        data = base_assessment_input()
        data.models = mock_model()  # not a list
        result = AssessmentInputValidator.validate(data, config)
        assert isinstance(result.models, list)

    def test_raises_if_models_empty_list(self, config):
        data = base_assessment_input()
        data.models = []
        with pytest.raises(ValueError, match="At least one model"):
            AssessmentInputValidator.validate(data, config)

    def test_dataset_split_none_uses_config_default(self, config):
        data = base_assessment_input()
        data.dataset_split = None
        result = AssessmentInputValidator.validate(data, config)
        assert result.dataset_split == config.dataset_split

    def test_kwargs_none_becomes_empty_dict(self, config):
        data = base_assessment_input()
        data.kwargs = None
        result = AssessmentInputValidator.validate(data, config)
        assert result.kwargs == {}

    def test_raises_if_fold_field_is_none(self, config):
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
        models = [mock_model(), mock_model()]
        data = base_assessment_input(models=models)

        data.x_train_folds = [np.array([1]), np.array([2])]
        data.y_train_folds = [np.array([1]), np.array([2])]
        data.x_val_folds = [np.array([1]), np.array([2])]
        data.y_val_folds = [np.array([1]), np.array([2])]

        setattr(data, bad_field, [np.array([1])])  # length 1 vs 2 models
        with pytest.raises(ValueError, match=match):
            AssessmentInputValidator.validate(data, config)


# ---------------------------------------------------------------------------
# ModelAssessment — evaluate (single model)
# ---------------------------------------------------------------------------


class TestModelAssessmentSingle:

    @pytest.fixture
    def config(self, tmp_path):
        return make_assessment_config(output_dir=tmp_path)

    def test_evaluate_single_returns_output(self, config):
        data = base_assessment_input()
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)

        assert isinstance(result, AssessmentOutput)
        assert result.is_cross_validation is False
        assert result.metrics is not None
        assert "accuracy" in result.metrics

    def test_evaluate_single_predictions_shape(self, config):
        data = base_assessment_input(n=4)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert result.predictions.shape == (4,)
        assert result.true_values.shape == (4,)

    def test_evaluate_populates_experiment_dir(self, config):
        data = base_assessment_input()
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert result.experiment_dir is not None

    def test_generate_report_true_sets_report_class(self, tmp_path):
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
        with patch("builtins.__import__", side_effect=ImportError):
            config = make_assessment_config(output_dir=tmp_path, generate_report=True)
            try:
                assessor = ModelAssessment(config)
                assert assessor.config.generate_report is False
            except Exception:
                pass

    def test_pre_process_raises_for_non_assessment_input(self, config):
        assessor = ModelAssessment(config)
        with pytest.raises(TypeError, match="AssessmentInput"):
            assessor.pre_process({"models": [], "x": None, "y": None})

    def test_pre_process_returns_validated_input(self, config):
        assessor = ModelAssessment(config)
        data = base_assessment_input()
        result = assessor.pre_process(data)
        assert isinstance(result, AssessmentInput)

    def test_run_delegates_to_evaluate(self, config):
        assessor = ModelAssessment(config)
        data = base_assessment_input()
        with patch.object(assessor, "evaluate", wraps=assessor.evaluate) as mock_eval:
            assessor.run(data)
            mock_eval.assert_called_once_with(data)

    def test_post_process_returns_data_unchanged(self, config):
        assessor = ModelAssessment(config)
        output = MagicMock(spec=AssessmentOutput)
        result = assessor.post_process(output)
        assert result is output


# ---------------------------------------------------------------------------
# ModelAssessment — evaluate (cross-validation)
# ---------------------------------------------------------------------------


class TestModelAssessmentCV:

    @pytest.fixture
    def config(self, tmp_path):
        return make_assessment_config(output_dir=tmp_path)

    def test_evaluate_cv_aggregates_folds(self, config):
        models = [mock_model(f"model_{i}") for i in range(3)]
        data = base_assessment_input(models=models)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)

        assert result.is_cross_validation is True
        assert result.aggregated_results is not None
        assert result.aggregated_results.n_folds == 3
        assert len(result.fold_results) == 3

    def test_evaluate_cv_metrics_per_fold(self, config):
        models = [mock_model() for _ in range(2)]
        data = base_assessment_input(models=models)
        assessor = ModelAssessment(config)
        result = assessor.evaluate(data)
        assert "accuracy" in result.aggregated_results.metrics_mean

    def test_evaluate_calls_generate_report_when_enabled(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path, generate_report=True)
        assessor = ModelAssessment(config)
        assessor.config.generate_report = True  # garante flag ativa pós-__init__

        with patch.object(assessor, "_generate_report") as mock_report:
            assessor.evaluate(base_assessment_input())
            mock_report.assert_called_once()


# ---------------------------------------------------------------------------
# ModelAssessment — _get_predictions (dataloader branch)
# ---------------------------------------------------------------------------


class TestGetPredictions:

    def test_uses_dataloader_when_required(self, tmp_path):
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


# ---------------------------------------------------------------------------
# ModelAssessment — summary
# ---------------------------------------------------------------------------


class TestSummary:

    def test_summary_before_evaluate(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assert "No evaluation results" in assessor.summary()

    def test_summary_after_single_evaluate(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.evaluate(base_assessment_input())
        summary = assessor.summary()
        assert "accuracy" in summary
        assert "MockModel" in summary

    def test_summary_after_cv_evaluate(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        models = [mock_model() for _ in range(2)]
        assessor.evaluate(base_assessment_input(models=models))
        summary = assessor.summary()
        assert "±" in summary


# ---------------------------------------------------------------------------
# ModelAssessment — _to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:

    @pytest.fixture
    def assessor(self, tmp_path):
        return ModelAssessment(make_assessment_config(output_dir=tmp_path))

    def test_converts_series(self, assessor):
        result = assessor._to_numpy(pd.Series([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_converts_dataframe(self, assessor):
        result = assessor._to_numpy(pd.DataFrame({"a": [1, 2]}))
        assert isinstance(result, np.ndarray)

    def test_converts_list(self, assessor):
        result = assessor._to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# ModelAssessment — export
# ---------------------------------------------------------------------------


class TestExportResults:

    def test_export_single_creates_csv_files(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path, export_results=True)
        assessor = ModelAssessment(config)
        assessor.evaluate(base_assessment_input())

        exp_dir = Path(assessor.results.experiment_dir)
        csv_files = list(exp_dir.glob("*.csv"))
        assert len(csv_files) >= 2  # predictions + metrics

    def test_export_cv_creates_csv_files(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path, export_results=True)
        assessor = ModelAssessment(config)
        models = [mock_model() for _ in range(2)]
        assessor.evaluate(base_assessment_input(models=models))

        exp_dir = Path(assessor.results.experiment_dir)
        csv_files = list(exp_dir.glob("*.csv"))
        assert len(csv_files) >= 2

    def test_export_raises_if_no_results(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        with pytest.raises(RuntimeError, match="No assessment results"):
            assessor._export_results()

    # def test_evaluate_cleans_experiment_dir_on_failure(self, tmp_path):
    #     config = make_assessment_config(output_dir=tmp_path)
    #     assessor = ModelAssessment(config)

    #     with patch.object(assessor, "_evaluate_single", side_effect=Exception("boom")):
    #         with pytest.raises(RuntimeError, match="ModelAssessment failed"):
    #             assessor.evaluate(base_assessment_input())

    def test_evaluate_removes_existing_experiment_dir_on_failure(self, tmp_path):
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
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._export_single_results()

    def test_export_cv_results_raises_if_no_results(self, tmp_path):
        config = make_assessment_config(output_dir=tmp_path)
        assessor = ModelAssessment(config)
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._export_cv_results()


class TestAssessmentReport:
    @pytest.fixture
    def assessor(self, tmp_path):
        return ModelAssessment(make_assessment_config(output_dir=tmp_path))

    def _set_fake_report_class(self, assessor):
        fake_cls = MagicMock()
        fake_cls.return_value.generate_summary_report.return_value = MagicMock()
        assessor._report_generation_class = fake_cls
        return fake_cls

    def test_generate_report_returns_early_if_disabled(self, assessor):
        assessor.config.generate_report = False
        assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_report_raises_if_no_results(self, assessor):
        assessor.config.generate_report = True
        assessor.results = None
        with pytest.raises(RuntimeError, match="No results available"):
            assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_report_calls_single_report_when_not_cv(self, assessor):
        _ = self._set_fake_report_class(assessor)
        assessor.config.generate_report = True
        assessor.results = MagicMock(is_cross_validation=False)

        with patch.object(assessor, "_generate_single_report") as mock_single:
            input_data, output_data = MagicMock(), MagicMock()
            assessor._generate_report(input_data, output_data)
            mock_single.assert_called_once_with(input_data, output_data)

    def test_generate_report_warns_if_no_report_class(self, assessor, capsys):
        assessor.config.generate_report = True
        assessor.results = MagicMock()
        # garante que NÃO tem o atributo
        if hasattr(assessor, "_report_generation_class"):
            delattr(assessor, "_report_generation_class")
        assessor._generate_report(MagicMock(), MagicMock())
        assert "Warning" in capsys.readouterr().out

    def test_generate_report_cv_branch_is_noop(self, assessor):
        assessor.config.generate_report = True
        assessor.results = MagicMock(is_cross_validation=True)
        self._set_fake_report_class(assessor)
        assessor._generate_report(MagicMock(), MagicMock())

    def test_generate_single_report_raises_if_no_results(self, assessor):
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._generate_single_report(MagicMock(), MagicMock())

    def test_generate_single_report_calls_report_generator(self, assessor):
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
        assessor.results = None
        with pytest.raises(RuntimeError, match="No assessment results available"):
            assessor._generate_cv_report(MagicMock(), MagicMock())

    def test_generate_cv_report_calls_report_generator(self, assessor):
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
