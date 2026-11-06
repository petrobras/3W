"""
Data preparation for the unsupervised learning notebook (notebook 3).

Reads the windows saved by notebook 1, keeps the normal windows for training
and the normal and anomaly windows for testing, and converts them to the
float32 arrays that the one-class SVM and the LSTM autoencoder use.
"""

import os

import numpy as np


class UnsupervisedDataLoader:
    """Per-fold loader of normal (training) and normal/anomaly (test) windows."""

    def __init__(self, persistence, config):
        self.persistence = persistence
        self.config = config
        self.windowed_dir = os.path.join(persistence.cv_splits_dir, "windowed")

    def _load_split(self, fold_path, split):
        """Windows and classes of the `train` or `test` split of a fold."""
        path = os.path.join(fold_path, f"{split}_windowed.{self.config.SAVE_FORMAT}")
        if not os.path.exists(path):
            print(f"  {split}(not found)", end="")
            return [], []
        dfs, classes = self.persistence._load_dataframes(path, self.config.SAVE_FORMAT)
        print(f"  {split}({len(dfs)})", end="")
        return dfs, classes

    def load_per_fold_data(
        self,
        selected_anomaly_classes=[3, 4, 8],
        max_normal_samples=500,
        max_anomaly_samples=200,
    ):
        """
        Load data organized by fold for per-fold evaluation.

        Returns:
            dict: {fold_name: {'train_normal': arrays, 'test_normal': arrays,
                   'test_anomaly': arrays, 'test_anomaly_classes': classes}}
        """
        # Seeded generator: the same windows are sampled on every run
        rng = np.random.default_rng(42)

        if not os.path.exists(self.windowed_dir):
            raise FileNotFoundError(
                f"Windowed data directory not found: {self.windowed_dir}. "
                "Please run Data Treatment notebook first."
            )
        fold_dirs = sorted(
            d
            for d in os.listdir(self.windowed_dir)
            if d.startswith("fold_")
            and os.path.isdir(os.path.join(self.windowed_dir, d))
        )
        if not fold_dirs:
            raise FileNotFoundError("No fold directories found in windowed data.")

        def sample(windows, limit, classes=None):
            """Random subset of at most `limit` windows (and of their classes)."""
            if limit and len(windows) > limit:
                chosen = rng.choice(len(windows), limit, replace=False)
                windows = [windows[i] for i in chosen]
                if classes is not None:
                    classes = [classes[i] for i in chosen]
            return windows, classes

        fold_data = {}
        for fold_name in fold_dirs:
            fold_path = os.path.join(self.windowed_dir, fold_name)
            try:
                # Training: normal windows only. Test: normal and the selected anomalies
                train_dfs, train_classes = self._load_split(fold_path, "train")
                train_normal = [
                    df for df, cls in zip(train_dfs, train_classes) if int(cls) == 0
                ]
                test_dfs, test_classes = self._load_split(fold_path, "test")
                test_normal, test_anomaly, anomaly_classes = [], [], []
                for df, cls in zip(test_dfs, test_classes):
                    if int(cls) == 0:
                        test_normal.append(df)
                    elif int(cls) in selected_anomaly_classes:
                        test_anomaly.append(df)
                        anomaly_classes.append(cls)

                train_normal, _ = sample(train_normal, max_normal_samples)
                test_normal, _ = sample(test_normal, max_normal_samples // 2)
                test_anomaly, anomaly_classes = sample(
                    test_anomaly, max_anomaly_samples, anomaly_classes
                )

                if len(train_normal) > 0 and (
                    len(test_normal) > 0 or len(test_anomaly) > 0
                ):
                    train_scaled, test_scaled, data_info, test_class_labels = (
                        prepare_arrays(
                            train_normal,
                            test_normal + test_anomaly,
                            [0] * len(test_normal) + anomaly_classes,
                        )
                    )
                    n_test_normal = len(test_normal)
                    fold_data[fold_name] = {
                        "train_normal": train_scaled,
                        "test_normal": test_scaled[:n_test_normal],
                        "test_anomaly": test_scaled[n_test_normal:],
                        "test_anomaly_classes": test_class_labels[n_test_normal:]
                        if test_class_labels
                        else [],
                        "data_info": data_info,
                    }
            except Exception as e:
                print(f"Error loading {fold_name}: {str(e)}")

        return fold_data


def _to_arrays(windows):
    """Sensor values of each window (label column removed), keeping the finite ones."""
    arrays = []
    for window in windows:
        values = window.drop(columns="class", errors="ignore").values
        if np.isfinite(values).all():
            arrays.append(values)
    return arrays


def prepare_arrays(train_windows, test_windows, test_classes):
    """
    Convert window dataframes to arrays of shape (windows, time steps, sensors).

    The autoencoder has a sigmoid output, so its inputs must lie in [0, 1]. The
    upstream scaling is z-score-like (toolkit `Normalize`): each sensor is
    rescaled with the range of the normal (training) windows only, and the
    values are clipped (anomalies may fall outside the normal range).

    Returns:
        tuple: (train_scaled, test_scaled, data_info, test_classes)
    """
    normal_arrays = _to_arrays(train_windows)
    test_arrays = _to_arrays(test_windows)
    if not normal_arrays or not test_arrays:
        raise ValueError("No valid arrays found after conversion")

    shape = normal_arrays[0].shape  # keep only windows with the same shape
    normal = np.array([a for a in normal_arrays if a.shape == shape], dtype=np.float32)
    test = np.array([a for a in test_arrays if a.shape == shape], dtype=np.float32)

    low = normal.min(axis=(0, 1), keepdims=True)
    span = np.maximum(normal.max(axis=(0, 1), keepdims=True) - low, 1e-8)
    normal_scaled = np.clip((normal - low) / span, 0.001, 0.999)
    test_scaled = np.clip((test - low) / span, 0.001, 0.999)

    data_info = {
        "time_steps": normal_scaled.shape[1],
        "n_features": normal_scaled.shape[2],
        "n_normal_samples": normal_scaled.shape[0],
        "n_anomaly_samples": test_scaled.shape[0],
    }
    if test_classes is not None and len(test_classes) != test_scaled.shape[0]:
        test_classes = test_classes[: test_scaled.shape[0]]
    return normal_scaled, test_scaled, data_info, test_classes
