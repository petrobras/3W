"""Unit tests for ThreeWToolkit.pipeline (Pipeline and PipelineConfig)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ThreeWToolkit.core.base_dataset import BaseDatasetConfig
from ThreeWToolkit.core.base_trainer import BaseTrainerConfig
from ThreeWToolkit.core.base_transform import BaseTransformConfig
from ThreeWToolkit.pipeline import Pipeline, PipelineConfig


# --------------------------------------------------------------------- #
# Minimal concrete configs (the real ones are not needed to test the
# orchestration logic, and unit tests must not touch the real dataset).
# --------------------------------------------------------------------- #
class _DummyDatasetConfig(BaseDatasetConfig):
    """Minimal dataset config for tests."""

    _target: type = MagicMock


class _DummyTrainerConfig(BaseTrainerConfig):
    """Minimal trainer config for tests."""

    _target: type = MagicMock


class _DummyTransformConfig(BaseTransformConfig):
    """Minimal transform config for tests."""

    _target: type = MagicMock


def _make_config(**overrides) -> PipelineConfig:
    """Build a PipelineConfig with dummy required configs."""
    params = {
        "train_dataset_config": _DummyDatasetConfig(),
        "trainer_config": _DummyTrainerConfig(),
    }
    params.update(overrides)
    return PipelineConfig(**params)


def _fake_config(**overrides) -> SimpleNamespace:
    """Config whose components are built from MagicMocks."""

    def _component_config(component):
        """Wrap a component in a config-like object."""
        return SimpleNamespace(build=MagicMock(return_value=component))

    values = {
        "task": "training",
        "num_folds": None,
        "stratify_by": [],
        "save_results": False,
        "experiment_name": "exp",
        "trainer_config": _component_config(MagicMock(name="trainer")),
        "train_dataset_config": _component_config(MagicMock(name="train_ds")),
        "val_dataset_config": None,
        "test_dataset_config": None,
        "transform_config": None,
        "pre_transform_config": None,
        "assessment_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def pipeline() -> Pipeline:
    """Pipeline with only trainer and train dataset."""
    with patch("ThreeWToolkit.pipeline.ModelRecorder"):
        return Pipeline(_fake_config())  # type: ignore[arg-type]


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_defaults(self) -> None:
        """Check the default values of the config."""
        config = _make_config()
        assert config.task == "training"
        assert config.num_folds is None
        assert config.stratify_by == []
        assert config.save_results is False
        assert config.experiment_name == "experiment"
        assert config.test_dataset_config is None
        assert config.val_dataset_config is None
        assert config.assessment_config is None

    def test_target_is_pipeline(self) -> None:
        """Check the config targets the Pipeline class."""
        assert _make_config()._target is Pipeline

    def test_cross_validation_requires_num_folds(self) -> None:
        """Reject explicit None num_folds for cross-validation."""
        # The validator only runs when num_folds is passed explicitly
        # (Pydantic skips validation of omitted defaults), so passing
        # None is what triggers the error.
        with pytest.raises(ValidationError, match="num_folds must be provided"):
            _make_config(task="cross_validation", num_folds=None)

    def test_cross_validation_omitted_num_folds(self) -> None:
        """Reject an omitted num_folds for cross-validation."""
        with pytest.raises(ValidationError, match="num_folds must be provided"):
            _make_config(task="cross_validation")

    def test_cross_validation_with_num_folds(self) -> None:
        """Accept num_folds for cross-validation."""
        config = _make_config(task="cross_validation", num_folds=3)
        assert config.num_folds == 3

    def test_training_rejects_num_folds(self) -> None:
        """Reject num_folds when the task is training."""
        with pytest.raises(ValidationError, match="should not be provided"):
            _make_config(task="training", num_folds=3)

    @pytest.mark.parametrize("folds", [1, 0, -2])
    def test_num_folds_must_be_greater_than_one(self, folds: int) -> None:
        """Reject num_folds values not greater than one."""
        with pytest.raises(ValidationError):
            _make_config(task="cross_validation", num_folds=folds)

    def test_invalid_task(self) -> None:
        """Reject an unknown task."""
        with pytest.raises(ValidationError):
            _make_config(task="bogus")

    def test_missing_required_configs(self) -> None:
        """Reject a config without required fields."""
        with pytest.raises(ValidationError):
            PipelineConfig()  # type: ignore[call-arg]


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_builds_required_components_only(self) -> None:
        """Build only the required components."""
        config = _fake_config()
        with patch("ThreeWToolkit.pipeline.ModelRecorder") as recorder:
            pipe = Pipeline(config)  # type: ignore[arg-type]

        config.trainer_config.build.assert_called_once()
        config.train_dataset_config.build.assert_called_once()
        assert pipe.trainer is config.trainer_config.build.return_value
        assert pipe.train_dataset is config.train_dataset_config.build.return_value
        assert pipe.val_dataset is None
        assert pipe.test_dataset is None
        assert pipe.transform is None
        assert pipe.pre_transform is None
        assert pipe.assessment is None
        assert pipe.model_recorder is recorder.return_value

    def test_builds_optional_components(self) -> None:
        """Build the optional components when configured."""

        def cfg() -> SimpleNamespace:
            """Create a config-like object with a mocked build()."""
            return SimpleNamespace(build=MagicMock(return_value=MagicMock()))

        config = _fake_config(
            val_dataset_config=cfg(),
            test_dataset_config=cfg(),
            transform_config=cfg(),
            pre_transform_config=cfg(),
            assessment_config=cfg(),
        )
        with patch("ThreeWToolkit.pipeline.ModelRecorder"):
            pipe = Pipeline(config)  # type: ignore[arg-type]

        assert pipe.val_dataset is config.val_dataset_config.build.return_value
        assert pipe.test_dataset is config.test_dataset_config.build.return_value
        assert pipe.transform is config.transform_config.build.return_value
        assert pipe.pre_transform is config.pre_transform_config.build.return_value
        assert pipe.assessment is config.assessment_config.build.return_value

    def test_built_from_pipeline_config(self) -> None:
        """PipelineConfig.build() instantiates a Pipeline."""
        trainer, dataset = MagicMock(), MagicMock()
        with (
            patch.object(_DummyTrainerConfig, "build", return_value=trainer),
            patch.object(_DummyDatasetConfig, "build", return_value=dataset),
            patch("ThreeWToolkit.pipeline.ModelRecorder"),
        ):
            pipe = _make_config(experiment_name="abc").build()

        assert isinstance(pipe, Pipeline)
        assert pipe.trainer is trainer
        assert pipe.train_dataset is dataset


class TestPrepareData:
    """Tests for Pipeline._prepare_data."""

    def test_none_dataset_returns_none(self, pipeline: Pipeline) -> None:
        """Return None when the dataset is None."""
        assert pipeline._prepare_data(None) is None

    def test_without_transform_returns_dataset(self, pipeline: Pipeline) -> None:
        """Return the dataset untouched without a transform."""
        dataset = MagicMock()
        assert pipeline._prepare_data(dataset, fit=True) is dataset

    def test_transform_without_fit(self, pipeline: Pipeline) -> None:
        """Transform without fitting when fit is False."""
        pipeline.transform = MagicMock()
        dataset = MagicMock()

        out = pipeline._prepare_data(dataset, fit=False)

        pipeline.transform.fit.assert_not_called()
        pipeline.transform.transform.assert_called_once_with(dataset)
        assert out is pipeline.transform.transform.return_value

    def test_transform_with_fit(self, pipeline: Pipeline) -> None:
        """Fit and then transform when fit is True."""
        pipeline.transform = MagicMock()
        dataset = MagicMock()

        pipeline._prepare_data(dataset, fit=True)

        pipeline.transform.fit.assert_called_once_with(dataset)
        pipeline.transform.transform.assert_called_once_with(dataset)


class TestPreProcessData:
    """Tests for Pipeline.pre_process_data."""

    def test_skipped_without_pre_transform(self, pipeline: Pipeline) -> None:
        """Skip preprocessing without a pre-transform."""
        original = pipeline.train_dataset
        assert pipeline.pre_process_data() is None
        assert pipeline.train_dataset is original

    def test_only_train_dataset(self, pipeline: Pipeline) -> None:
        """Fit and apply the pre-transform on the train set."""
        original = pipeline.train_dataset
        pipeline.pre_transform = MagicMock()

        pipeline.pre_process_data()

        pipeline.pre_transform.fit.assert_called_once_with(original)
        pipeline.pre_transform.transform.assert_called_once_with(original)
        assert pipeline.train_dataset is pipeline.pre_transform.transform.return_value

    def test_val_and_test_transformed_but_not_fitted(self, pipeline: Pipeline) -> None:
        """Transform val and test sets without fitting."""
        train, val, test = MagicMock(), MagicMock(), MagicMock()
        pipeline.train_dataset, pipeline.val_dataset = train, val
        pipeline.test_dataset = test
        pipeline.pre_transform = MagicMock()
        pipeline.pre_transform.transform.side_effect = lambda ds: f"t_{id(ds)}"

        pipeline.pre_process_data()

        pipeline.pre_transform.fit.assert_called_once_with(train)
        assert pipeline.pre_transform.transform.call_count == 3
        assert pipeline.train_dataset == f"t_{id(train)}"
        assert pipeline.val_dataset == f"t_{id(val)}"
        assert pipeline.test_dataset == f"t_{id(test)}"


class TestTrain:
    """Tests for Pipeline.train."""

    def test_train_without_val(self, pipeline: Pipeline) -> None:
        """Train without a validation dataset."""
        train = MagicMock()

        result = pipeline.train(train, None)

        pipeline.trainer.train.assert_called_once_with(train, None)  # type: ignore[attr-defined]
        assert result is pipeline.trainer.train.return_value  # type: ignore[attr-defined]

    def test_train_fits_on_train_only(self, pipeline: Pipeline) -> None:
        """Fit the transform on the train set only."""
        pipeline.transform = MagicMock()
        pipeline.transform.transform.side_effect = lambda ds: ("t", ds)
        train, val = MagicMock(), MagicMock()

        pipeline.train(train, val)

        pipeline.transform.fit.assert_called_once_with(train)
        pipeline.trainer.train.assert_called_once_with(  # type: ignore[attr-defined]
            ("t", train), ("t", val)
        )

    def test_train_raises_when_data_not_prepared(self, pipeline: Pipeline) -> None:
        """Raise when the training data is not prepared."""
        with patch.object(pipeline, "_prepare_data", return_value=None):
            with pytest.raises(RuntimeError, match="prepare training data"):
                pipeline.train(MagicMock(), None)


class TestPredict:
    """Tests for Pipeline.predict."""

    def test_none_dataset_skips_prediction(self, pipeline: Pipeline) -> None:
        """Skip prediction when there is no dataset."""
        assert pipeline.predict(None) is None
        pipeline.trainer.predict.assert_not_called()  # type: ignore[attr-defined]

    def test_predict_uses_trainer(self, pipeline: Pipeline) -> None:
        """Delegate prediction to the trainer."""
        dataset = MagicMock()
        result = pipeline.predict(dataset)
        pipeline.trainer.predict.assert_called_once_with(dataset)  # type: ignore[attr-defined]
        assert result is pipeline.trainer.predict.return_value  # type: ignore[attr-defined]

    def test_predict_applies_transform_without_fit(self, pipeline: Pipeline) -> None:
        """Transform the data before predicting, no fit."""
        pipeline.transform = MagicMock()
        pipeline.transform.transform.return_value = "transformed"

        pipeline.predict(MagicMock())

        pipeline.transform.fit.assert_not_called()
        pipeline.trainer.predict.assert_called_once_with("transformed")  # type: ignore[attr-defined]


class TestAssess:
    """Tests for Pipeline.assess."""

    def test_without_assessment_returns_none(self, pipeline: Pipeline) -> None:
        """Return None without an assessment component."""
        assert pipeline.assess(MagicMock(), None) is None

    def test_with_assessment(self, pipeline: Pipeline) -> None:
        """Delegate evaluation to the assessment component."""
        pipeline.assessment = MagicMock()
        training, preds = MagicMock(), MagicMock()

        out = pipeline.assess(training, preds)

        pipeline.assessment.evaluate.assert_called_once_with(training, preds)
        assert out is pipeline.assessment.evaluate.return_value


class TestCrossValidate:
    """Tests for Pipeline.cross_validate."""

    def test_runs_one_training_per_fold(self, pipeline: Pipeline) -> None:
        """Train once per fold and collect the results."""
        folds = [(MagicMock(), MagicMock()) for _ in range(3)]
        pipeline.trainer.config.seed = 7  # type: ignore[attr-defined]
        with (
            patch("ThreeWToolkit.pipeline.KFoldSplitter") as splitter_cls,
            patch.object(pipeline, "train", side_effect=["r1", "r2", "r3"]) as train,
            patch("ThreeWToolkit.pipeline.CrossValidationResult") as cv_result,
        ):
            splitter_cls.return_value.split_data.return_value = iter(folds)
            out = pipeline.cross_validate(
                MagicMock(), 3, stratify_by=["a"], random_state=11
            )

        splitter_cls.assert_called_once_with(
            num_splits=3, stratify_by=["a"], random_state=11
        )
        assert [c.args for c in train.call_args_list] == folds
        kwargs = cv_result.call_args.kwargs
        assert kwargs["fold_results"] == ["r1", "r2", "r3"]
        assert kwargs["metadata"]["num_folds"] == 3
        assert kwargs["metadata"]["stratify_by"] == ["a"]
        assert kwargs["metadata"]["seed"] == 7
        assert kwargs["metadata"]["split_random_state"] == 11
        assert out is cv_result.return_value

    def test_stratify_by_none_becomes_empty_list(self, pipeline: Pipeline) -> None:
        """Convert a None stratify_by into an empty list."""
        with (
            patch("ThreeWToolkit.pipeline.KFoldSplitter") as splitter_cls,
            patch("ThreeWToolkit.pipeline.CrossValidationResult") as cv_result,
        ):
            splitter_cls.return_value.split_data.return_value = iter([])
            pipeline.cross_validate(MagicMock(), 2)

        assert splitter_cls.call_args.kwargs["stratify_by"] == []
        assert cv_result.call_args.kwargs["metadata"]["stratify_by"] == []
        assert cv_result.call_args.kwargs["fold_results"] == []


@pytest.fixture
def run_pipeline(pipeline: Pipeline):
    """Pipeline with every step of run() replaced by mocks."""
    pipeline.train_dataset = MagicMock()
    pipeline.train_dataset.__len__.return_value = 10
    pipeline.pre_process_data = MagicMock()  # type: ignore[method-assign]
    pipeline.train = MagicMock(return_value="train_res")  # type: ignore[method-assign]
    pipeline.cross_validate = MagicMock(return_value="cv_res")  # type: ignore[method-assign]
    pipeline.predict = MagicMock(return_value="preds")  # type: ignore[method-assign]
    pipeline.assess = MagicMock(return_value="assessment")  # type: ignore[method-assign]
    pipeline.trainer.config.seed = 5  # type: ignore[attr-defined]
    with patch("ThreeWToolkit.pipeline.PipelineResult") as result_cls:
        yield pipeline, result_cls


class TestRun:
    """Tests for Pipeline.run."""

    def test_training_task(self, run_pipeline) -> None:
        """Run the training task end to end."""
        pipe, result_cls = run_pipeline

        out = pipe.run()

        pipe.pre_process_data.assert_called_once()
        pipe.train.assert_called_once_with(pipe.train_dataset, None)
        pipe.cross_validate.assert_not_called()
        pipe.predict.assert_called_once_with(None)
        pipe.assess.assert_called_once_with("train_res", "preds")
        kwargs = result_cls.call_args.kwargs
        assert kwargs["training_result"] == "train_res"
        assert kwargs["assessment_output"] == "assessment"
        assert kwargs["experiment_name"] == "exp"
        assert kwargs["metadata"] == {
            "train_size": 10,
            "val_size": 0,
            "test_size": 0,
        }
        assert out is result_cls.return_value

    def test_metadata_with_val_and_test(self, run_pipeline) -> None:
        """Report val and test sizes in the metadata."""
        pipe, result_cls = run_pipeline
        pipe.val_dataset = MagicMock()
        pipe.val_dataset.__len__.return_value = 3
        pipe.test_dataset = MagicMock()
        pipe.test_dataset.__len__.return_value = 4

        pipe.run()

        pipe.train.assert_called_once_with(pipe.train_dataset, pipe.val_dataset)
        pipe.predict.assert_called_once_with(pipe.test_dataset)
        assert result_cls.call_args.kwargs["metadata"] == {
            "train_size": 10,
            "val_size": 3,
            "test_size": 4,
        }

    def test_cross_validation_task(self, run_pipeline) -> None:
        """Run the cross-validation task."""
        pipe, result_cls = run_pipeline
        pipe.config.task = "cross_validation"
        pipe.config.num_folds = 4
        pipe.config.stratify_by = ["col"]

        pipe.run()

        pipe.cross_validate.assert_called_once_with(
            pipe.train_dataset, 4, ["col"], random_state=5
        )
        pipe.train.assert_not_called()
        pipe.assess.assert_called_once_with("cv_res", "preds")

    def test_cross_validation_without_num_folds(self, run_pipeline) -> None:
        """Raise when num_folds is missing at run time."""
        pipe, _ = run_pipeline
        pipe.config.task = "cross_validation"
        pipe.config.num_folds = None

        with pytest.raises(ValueError, match="num_folds must be specified"):
            pipe.run()

    def test_unsupported_task(self, run_pipeline) -> None:
        """Raise for an unsupported task."""
        pipe, _ = run_pipeline
        pipe.config.task = "unknown"

        with pytest.raises(ValueError, match="Unsupported task"):
            pipe.run()

    def test_nothing_saved_by_default(self, run_pipeline) -> None:
        """Save nothing when save_results is False."""
        pipe, _ = run_pipeline
        pipe.run()
        pipe.model_recorder.save_model.assert_not_called()
        pipe.model_recorder.save_transform.assert_not_called()

    def test_save_model_only(self, run_pipeline) -> None:
        """Save only the model when there is no transform."""
        pipe, _ = run_pipeline
        pipe.config.save_results = True

        pipe.run()

        pipe.model_recorder.save_model.assert_called_once_with(
            pipe.trainer.model, "exp"
        )
        pipe.model_recorder.save_transform.assert_not_called()

    def test_save_model_and_transform(self, run_pipeline) -> None:
        """Save both the model and the transform."""
        pipe, _ = run_pipeline
        pipe.config.save_results = True
        pipe.transform = MagicMock()

        pipe.run()

        pipe.model_recorder.save_transform.assert_called_once_with(
            pipe.transform, "exp_transform.pkl"
        )
        pipe.model_recorder.save_model.assert_called_once_with(
            pipe.trainer.model, "exp"
        )
