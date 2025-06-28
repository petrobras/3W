import pytest
import pandas as pd
from ThreeWToolkit.holdout import TimeSeriesHoldout


class TestTimeSeriesHoldout:
    def setup_method(self):
        """
        Setup common test data.
        """
        self.df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [10, 20, 30, 40, 50]
        })
        self.config = {}

    def test_train_test_split_default(self):
        """
        Test basic split using internal self.data.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        train, test = ts.train_test_split(train_size=0.6)

        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) == 3
        assert len(test) == 2
        assert train.iloc[-1]["value"] == 30
        assert test.iloc[0]["value"] == 40

    def test_train_test_split_with_explicit_array(self):
        """
        Test split using an explicitly passed array.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        train, test = ts.train_test_split(self.df["value"], test_size=0.2)

        assert isinstance(train, pd.Series)
        assert isinstance(test, pd.Series)
        assert list(train) == [10, 20, 30, 40]
        assert list(test) == [50]

    def test_train_test_split_invalid_input_type(self):
        """
        Test passing invalid input types raises TypeError.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)

        with pytest.raises(TypeError, match="All inputs must be pandas Series or DataFrame."):
            ts.train_test_split([1, 2, 3], test_size=0.5)

    def test_train_test_split_raises_runtime_error(self, monkeypatch):
        """
        Force failure of sklearn_train_test_split to trigger RuntimeError.
        """
        import ThreeWToolkit.holdout.train_test_split as tts_module

        def broken_split(*args, **kwargs):
            raise ValueError("simulated failure")

        monkeypatch.setattr(tts_module, "sklearn_train_test_split", broken_split)

        ts = tts_module.TimeSeriesHoldout(data=self.df, pip_config=self.config)

        with pytest.raises(RuntimeError, match="Failed to split time series data: simulated failure"):
            ts.train_test_split(test_size=0.5)

    def test_train_test_split_test_size_gt_one(self):
        """
        Test that test_size > 1 raises ValueError.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        with pytest.raises(ValueError, match="test_size must be <= 1"):
            ts.train_test_split(test_size=1.5)

    def test_train_test_split_train_size_gt_one(self):
        """
        Test that train_size > 1 raises ValueError.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        with pytest.raises(ValueError, match="train_size must be <= 1"):
            ts.train_test_split(train_size=1.5)

    def test_train_test_split_sum_gt_one(self):
        """
        Test that train_size + test_size > 1 raises ValueError.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        with pytest.raises(ValueError, match="The sum of train_size and test_size"):
            ts.train_test_split(train_size=0.6, test_size=0.6)

    def test_train_test_split_sum_less_one(self):
        """
        Test that train_size + test_size < 1.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        ts.train_test_split(train_size=0.7, test_size=0.3)

    def test_train_test_split_stratify_requires_shuffle(self):
        """
        Test that stratified splitting raises error if shuffle is False.
        """
        ts = TimeSeriesHoldout(data=self.df, pip_config=self.config)
        with pytest.raises(ValueError, match="Stratified splitting requires shuffle=True."):
            ts.train_test_split(test_size=0.2, stratify=self.df["value"], shuffle=False)

    def test_train_test_split_stratify_ok_if_shuffle_true(self):
        """
        Test that stratified splitting works correctly when shuffle is True.
        """
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "value": [1, 1, 2, 2, 3, 3]  # Each class must appear at least 2 times.
        })

        config = {"shuffle": True, "stratify": df["value"], "test_size": 0.4}
        ts = TimeSeriesHoldout(data=df, pip_config=config)
        train, test = ts.train_test_split()
        assert len(train) + len(test) == 6

    def test_train_test_split_shuffle_fallback_from_config(self):
        """
        Test that shuffle=True from config is correctly used in splitting.
        """
        config = {"shuffle": True, "test_size": 0.4}
        ts = TimeSeriesHoldout(data=self.df, pip_config=config)
        train, test = ts.train_test_split()
        assert len(test) == 2