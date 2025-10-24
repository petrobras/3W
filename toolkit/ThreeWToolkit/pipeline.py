import torch
import logging
import pandas as pd
from typing import Any, Optional
from pydantic import BaseModel

from .models.mlp import MLP
from tqdm.auto import tqdm

from .assessment.model_assess import ModelAssessment
from .core.base_assessment import ModelAssessmentConfig
from .core.base_dataset import ParquetDatasetConfig
from .core.base_preprocessing import (
    ImputeMissingConfig,
    NormalizeConfig,
    RenameColumnsConfig,
    WindowingConfig,
)
from .core.base_step import BaseStep
from .dataset.parquet_dataset import ParquetDataset
from .feature_extraction.extract_exponential_statistics_features import (
    EWStatisticalConfig,
    ExtractEWStatisticalFeatures,
)
from .feature_extraction.extract_statistical_features import (
    ExtractStatisticalFeatures,
    StatisticalConfig,
)
from .feature_extraction.extract_wavelet_features import (
    ExtractWaveletFeatures,
    WaveletConfig,
)
from .models.sklearn_models import SklearnModels
from .preprocessing._data_processing import (
    ImputeMissing,
    Normalize,
    RenameColumns,
    Windowing,
)
from .trainer.trainer import ModelTrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class Pipeline:
    """
    A flexible machine learning pipeline for processing time series data.

    This pipeline orchestrates the complete ML workflow from data loading through
    preprocessing, feature extraction, model training, and assessment. It supports
    various configurations for each step and ensures proper validation and ordering
    of processing steps.

    Attributes:
        step_data_loader (Optional[ParquetDataset]): Data loading component
        step_preprocessing (list[Any]): List of preprocessing steps
        step_feat_extraction (list[Any]): List of feature extraction steps
        step_model_training (Optional[ModelTrainer]): Model training component
        step_model_assessment (Optional[ModelAssessment]): Model assessment component
    """

    def __init__(self, configs: list[BaseModel]):
        """
        Initialize the pipeline with a list of configuration objects.

        Args:
            configs (list[BaseModel]): List of configuration objects for different
                                     pipeline steps (dataset, preprocessing, feature
                                     extraction, training, assessment)
        """
        # Initialize class attributes
        self.step_data_loader: Optional[ParquetDataset] = None
        self.step_preprocessing: list[Any] = []
        self.step_feat_extraction: list[Any] = []
        self.step_model_training: Optional[ModelTrainer] = None
        self.step_model_assessment: Optional[ModelAssessment] = None

        self._factory_model(configs)
        self._validate_steps()

    def _factory_model(self, configs: list[BaseModel]):
        """
        Factory method to instantiate pipeline components based on configurations.

        Processes each configuration object and creates the corresponding pipeline
        component (dataset loader, preprocessing steps, feature extractors, etc.).

        Args:
            configs (list[BaseModel]): List of configuration objects

        Raises:
            ValueError: If an unsupported configuration type is provided
        """
        for config in configs:
            # Dataset
            if isinstance(config, ParquetDatasetConfig):
                self.step_data_loader = ParquetDataset(config)
            # Preprocessing
            elif isinstance(config, ImputeMissingConfig):
                self.step_preprocessing.append(ImputeMissing(config))
            elif isinstance(config, NormalizeConfig):
                self.step_preprocessing.append(Normalize(config))
            elif isinstance(config, RenameColumnsConfig):
                self.step_preprocessing.append(RenameColumns(config))
            elif isinstance(config, WindowingConfig):
                self.step_preprocessing.append(Windowing(config))
            # Feature Extraction
            elif isinstance(config, StatisticalConfig):
                self.step_feat_extraction.append(ExtractStatisticalFeatures(config))
            elif isinstance(config, WaveletConfig):
                self.step_feat_extraction.append(ExtractWaveletFeatures(config))
            elif isinstance(config, EWStatisticalConfig):
                self.step_feat_extraction.append(ExtractEWStatisticalFeatures(config))
            # Model Training
            elif isinstance(config, TrainerConfig):
                self.step_model_training = ModelTrainer(config)
            # Model Assessment
            elif isinstance(config, ModelAssessmentConfig):
                self.step_model_assessment = ModelAssessment(config)
            else:
                raise ValueError(f"Configuration {config} is not supported")

    def _validate_steps(self):
        """
        Validate that all mandatory pipeline steps are properly defined.

        Checks for required components and ensures they inherit from BaseStep.
        Also validates and fixes the order of preprocessing steps.

        Raises:
            ValueError: If mandatory steps are missing or improperly configured
            TypeError: If steps don't inherit from BaseStep
        """
        # Check if mandatory steps are defined
        if self.step_data_loader is None:
            raise ValueError("The 'data_loader' step must be defined.")

        if not isinstance(self.step_data_loader, BaseStep):
            raise TypeError("The 'data_loader' step must inherit from BaseStep.")

        self.step_preprocessing = self._validate_and_fix_preprocessing_steps(
            self.step_preprocessing
        )

        if len(self.step_feat_extraction) > 1:
            raise ValueError(
                f"Only one feature extraction strategy is allowed, "
                f"but {len(self.step_feat_extraction)} were provided."
            )

        for feat_extraction in self.step_feat_extraction:
            if not isinstance(feat_extraction, BaseStep):
                raise TypeError(
                    f"The step '{feat_extraction.__class__.__name__}' must inherit from BaseStep."
                )

        if not isinstance(self.step_model_training, BaseStep):
            raise ValueError("The 'model_training' step must inherit from BaseStep.")
        else:
            if not isinstance(self.step_model_training, (ModelTrainer, SklearnModels)):
                raise ValueError(
                    "The 'model_training' step must be a ModelTrainer or SklearnModels instance."
                )

        if not isinstance(self.step_model_assessment, BaseStep):
            raise ValueError("The 'model_assessment' step must inherit from BaseStep.")

    def _validate_and_fix_preprocessing_steps(self, steps):
        """
        Validate and automatically adjust the order of preprocessing steps.

        Enforces the following rules:
        - Only one RenameColumns step is allowed
        - If Windowing exists, it must always be the last step
        - If Windowing does not exist, RenameColumns must be the last step

        Args:
            steps (list): List of preprocessing steps

        Returns:
            list: Reordered list of preprocessing steps

        Raises:
            TypeError: If steps don't inherit from BaseStep
            ValueError: If multiple RenameColumns steps are provided
        """
        for step in steps:
            if not isinstance(step, BaseStep):
                raise TypeError(
                    f"Step '{step.__class__.__name__}' must inherit from BaseStep."
                )

        rename_steps = [s for s in steps if isinstance(s, RenameColumns)]
        windowing_steps = [s for s in steps if isinstance(s, Windowing)]

        # More than one RenameColumns is not allowed
        if len(rename_steps) > 1:
            raise ValueError("Only one RenameColumns step is allowed.")

        # Adjust order when Windowing exists
        if windowing_steps:
            # If Windowing is not the last step, move it
            if not isinstance(steps[-1], Windowing):
                logging.info("[Pipeline] Moving Windowing step to the last position.")
                steps = [
                    s for s in steps if not isinstance(s, Windowing)
                ] + windowing_steps

            # If RenameColumns exists after Windowing, move it before
            if rename_steps and steps.index(rename_steps[0]) > steps.index(
                windowing_steps[0]
            ):
                logging.info("[Pipeline] Moving RenameColumns step before Windowing.")
                steps.remove(rename_steps[0])
                idx_win = steps.index(windowing_steps[0])
                steps.insert(idx_win, rename_steps[0])

        else:
            # No Windowing → RenameColumns must be the last step
            if rename_steps and not isinstance(steps[-1], RenameColumns):
                logging.info(
                    "[Pipeline] Moving RenameColumns step to the last position."
                )
                steps.remove(rename_steps[0])
                steps.append(rename_steps[0])

        return steps

    def _batch_iterator(self, data_loader, feature_names=None):
        """
        Iterator for processing batches from DataLoader or list of batches.

        Converts batch data to DataFrame format with X features and y target,
        where y is the last column.

        Args:
            data_loader: DataLoader or iterable containing batch data
            feature_names (list, optional): Names for feature columns

        Yields:
            pd.DataFrame: DataFrame with features and target (if available)

        Raises:
            ValueError: If batch format is not supported
        """
        if data_loader is not None:
            for batch in data_loader:
                if isinstance(batch, torch.Tensor):
                    x, y = batch, None
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                # Convert X to DataFrame
                x_df = pd.DataFrame(
                    x.numpy() if isinstance(x, torch.Tensor) else x,
                    columns=feature_names,
                )

                # Convert y to Series
                y_series = (
                    pd.Series(
                        y.numpy().ravel() if isinstance(y, torch.Tensor) else y,
                        name="target",  # fixed or configurable name
                    )
                    if y is not None
                    else None
                )

                # Concatenate X and y (if it exists)
                if y_series is not None:
                    df = pd.concat([x_df, y_series], axis=1)
                else:
                    df = x_df

                yield df

    def run(self):
        """
        Execute the complete machine learning pipeline.

        Processes all data batches through preprocessing and feature extraction,
        trains the model, and performs assessment. Displays progress using rich
        progress bars.

        Raises:
            NotImplementedError: If model type is not supported or training is undefined
        """
        # Train the model if training step exists
        if self.step_model_training:
            if not isinstance(self.step_model_training.model, (MLP, SklearnModels)):
                raise NotImplementedError("Model not implemented yet.")

            if self.step_data_loader is None:
                raise ValueError("Data loader step is not defined.")

            total_files = len(self.step_data_loader)

            dfs = []

            with tqdm(
                total=total_files,
                desc="[Pipeline] Processing batches",
                unit="file",
                colour="#386641",
            ) as pbar:
                for batch in self.step_data_loader.iterbatches():
                    nfiles = self._get_nfiles_per_batch(batch)
                    df_processed = self.run_prep_steps(batch)
                    dfs.append(df_processed)

                    pbar.update(nfiles)

            dfs_final = pd.concat(dfs, ignore_index=True, axis=0)

            x_train, y_train, x_test, y_test = self.step_model_training.holdout(
                X=dfs_final.iloc[:, :-1],
                Y=dfs_final["label"].astype(int),
                test_size=self.step_model_training.test_size,
            )
            results = self.step_model_training((x_train, y_train))

            if self.step_model_assessment is None:
                raise ValueError("Model assessment step is not defined.")

            self.step_model_assessment((results["model"], x_test, y_test))
        else:
            raise NotImplementedError("TODO: Implement pipeline without training")

    def run_prep_steps(self, batch: dict) -> pd.DataFrame | None:
        """
        Execute preprocessing and feature extraction steps on a batch.

        Sequentially applies all preprocessing steps followed by feature extraction
        to prepare the data for model training.

        Args:
            batch: Input batch data

        Returns:
            pd.DataFrame: Processed batch ready for training
        """
        # Execute all preprocessing steps
        batch_prep = batch

        if self.step_preprocessing:
            batch_prep = self.run_step_preprocessing(batch_prep)

        # Execute feature extraction
        if self.step_feat_extraction:
            batch_prep = self.run_step_feature_extraction(batch_prep)
        else:
            if isinstance(batch_prep, dict):
                _, batch_prep = self._check_and_apply_windowing(batch_prep)

        return batch_prep

    def run_step_preprocessing(self, batch: dict) -> dict[Any, Any]:
        """
        Execute preprocessing steps on the batch data.

        Applies each preprocessing step sequentially to all files in the batch.
        Each step receives a DataFrame with all signals for a file and returns
        a processed DataFrame that may have different columns (renamed, created, or removed).

        Args:
            batch (dict): Batch containing 'signals' and 'labels' data

        Returns:
            dict: Processed batch with updated signals and labels structure
        """
        if self.step_data_loader is None:
            raise ValueError("Data loader step is not defined.")

        # Create a copy of the batch to avoid modifying the original
        batch_processed = {}
        batch_processed.update({key: value for key, value in list(batch.items())})

        # Progress bar for preprocessing steps
        for step in tqdm(
            self.step_preprocessing,
            desc="[Pipeline] Preprocessing steps",
            leave=False,
            unit="step",
            colour="#a7c957",
        ):
            keys = list(batch["signals"].keys())
            target_column = self.step_data_loader.config.target_column

            # Initialize new batch structure for processed data
            new_signals = {}
            new_labels = []

            nfiles = self._get_nfiles_per_batch(batch)

            # Process each file in the batch individually
            for idx_data_file in range(nfiles):
                file_labels = batch["labels"][target_column][idx_data_file]

                # Prepare DataFrames for each signal in the current file
                dfs_for_file = []
                for signal_name in keys:
                    signal_data = batch["signals"][signal_name][idx_data_file]
                    # Convert signal data to DataFrame with proper column name
                    df_signal = pd.DataFrame(signal_data, columns=[signal_name])
                    dfs_for_file.append(df_signal)

                # Concatenate all signals for this file into a single DataFrame
                df_file = pd.concat(dfs_for_file, axis=1)

                # Clean up: remove label column if it exists (shouldn't at this stage)
                if "label" in df_file.columns:
                    df_file = df_file.drop("label", axis=1)

                # Apply the preprocessing step to the file's data
                processed_df: pd.DataFrame = step(df_file)

                # Store processed data as dictionary for this file
                new_signals[idx_data_file] = processed_df.to_dict()

                if not step.__class__.__name__ == "Windowing":
                    new_labels.append(file_labels)
                else:
                    labels_series = pd.Series(file_labels)
                    label_step_config = step.config.copy()
                    label_step_config.window = "boxcar"
                    label_step_config.normalize = False
                    label_step = Windowing(label_step_config)

                    windowed_label = label_step(labels_series)
                    windowed_label.drop(columns=["win"], inplace=True)

                    new_labels.append(windowed_label.mode(axis=1)[0].values)

            # Update batch with processed signals and labels
            batch_processed["signals"] = new_signals
            batch_processed["labels"] = new_labels

        return batch_processed

    def _get_nfiles_per_batch(self, batch):
        """
        Get the number of files in a batch.

        Args:
            batch (dict): Batch containing signals data

        Returns:
            int: Number of files in the batch
        """
        signals = batch["signals"]
        nfiles = len(signals[next(iter(signals))])
        return nfiles

    def run_step_feature_extraction(self, batch: dict) -> pd.DataFrame:
        """
        Execute feature extraction steps on the preprocessed batch data.

        Converts windowed signal data into features by applying feature extraction
        algorithms. Ensures windowing is applied if not already done during preprocessing.

        Args:
            batch (dict): Preprocessed batch containing windowed signals and labels

        Returns:
            pd.DataFrame: Feature matrix with extracted features and labels
        """
        if self.step_data_loader is None:
            raise ValueError("Data loader step is not defined.")

        df = pd.DataFrame()
        # Progress bar for feature extraction steps
        for step in tqdm(
            self.step_feat_extraction,
            desc="[Pipeline] Feature Extraction",
            leave=False,
            unit="step",
            colour="#a7c957",
        ):
            # Check if windowing was applied during preprocessing
            is_windowing_applied, df_batch = self._check_and_apply_windowing(batch)

            # Configure the feature extraction step
            if isinstance(
                step, (ExtractEWStatisticalFeatures, ExtractStatisticalFeatures)
            ):
                if not is_windowing_applied:
                    window_size = WindowingConfig.window_size
                else:
                    cls: Windowing = next(
                        c
                        for c in self.step_preprocessing
                        if c.__class__.__name__ == "Windowing"
                    )
                    window_size = cls.config.window_size
                step.window_size = window_size
            step.is_windowed = True
            step.label_column = "label"

            # Apply feature extraction to the entire batch
            df = step(df_batch)

        return df

    def _check_and_apply_windowing(
        self, batch: dict[Any, Any]
    ) -> tuple[bool, pd.DataFrame]:
        """
        Ensure that windowing has been applied to the batch.

        If no instance of `Windowing` exists in the preprocessing steps,
        a default windowing configuration (`pad_last_window=True`) is applied.
        Each signal in the batch is then converted into a DataFrame,
        windowing is applied if required, and the data is prepared for
        feature extraction.

        Args:
            batch (dict): Dictionary containing:
                - "signals": list of arrays or DataFrames with raw signals
                - "labels": list of arrays with the corresponding labels

        Returns:
            tuple:
                - bool: whether windowing was already applied
                - pd.DataFrame: concatenated DataFrame containing all signals
                (with windowing applied if necessary) and the "label" column
        """
        is_windowing_applied = any(
            isinstance(s, Windowing) for s in self.step_preprocessing
        )
        if not is_windowing_applied:
            logging.warning(
                "Feature extraction classes require data windowing. None was provided; "
                "default windowing will be applied."
            )
            # Apply default windowing configuration
            windowing = Windowing(WindowingConfig(pad_last_window=True))
            label_windowing = Windowing(
                WindowingConfig(window="boxcar", pad_last_window=True)
            )

            all_dfs = []
            # Process each file in the batch
            for idx_data_file in range(len(batch["signals"])):
                # Get signal data for this file
                signal_data = batch["signals"][idx_data_file]

                file_labels = batch["labels"][idx_data_file]

                # Convert signal data to DataFrame
                df_signal = pd.DataFrame(signal_data)
                # Convert label data to Series
                labels_series = pd.Series(file_labels)

                df_signal = windowing(df_signal)
                df_label = label_windowing(labels_series)

                # Clean up: remove windowing index column if present
                df_signal = df_signal.drop("win", axis=1, errors="ignore")
                df_label = df_label.drop("win", axis=1, errors="ignore")

                df_signal["label"] = df_label.mode(axis=1)[0].values
                all_dfs.append(df_signal)

            # Concatenate all files into a single DataFrame for feature extraction
            df_batch = pd.concat(all_dfs, axis=0, ignore_index=True)

            return is_windowing_applied, df_batch

        else:
            all_dfs = []

            for idx_data_file in range(len(batch["signals"])):
                signal_data = batch["signals"][idx_data_file]

                df_signal = pd.DataFrame(signal_data)

                df_signal = df_signal.drop("win", axis=1, errors="ignore")

                df_signal["label"] = batch["labels"][idx_data_file]

                all_dfs.append(df_signal)

            # Concatenate all files into a single DataFrame for feature extraction
            df_batch = pd.concat(all_dfs, axis=0, ignore_index=True)

            return is_windowing_applied, df_batch
