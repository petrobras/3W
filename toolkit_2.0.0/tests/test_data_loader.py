import os
import pytest
import pandas as pd
from tempfile import NamedTemporaryFile

from ThreeWToolkit.data_loader import load_csv


class TestDataLoader:
    def test_load_csv_basic(self):
        """
        Test basic functionality of the load_csv function with correct inputs.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value\n2024-01-01,100\n2024-01-02,200")
            tmp_path = tmp.name

        column_names = ["date", "value"]
        date_column = ["date"]
        parse_dates = True

        try:
            df = load_csv(
                file_path=tmp_path,
                column_names=column_names,
                date_column=date_column,
                parse_dates=parse_dates,
            )

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == column_names
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df.shape == (2, 2)
            assert df.iloc[0]["value"] == 100
        finally:
            os.remove(tmp_path)

    def test_load_csv_no_date_parsing(self):
        """
        Test CSV loading when date parsing is disabled.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value\n2024-01-01,100")
            tmp_path = tmp.name

        try:
            df = load_csv(
                file_path=tmp_path,
                column_names=["date", "value"],
                date_column=["date"],
                parse_dates=False,
            )

            assert not pd.api.types.is_datetime64_any_dtype(df["date"])
        finally:
            os.remove(tmp_path)

    def test_load_csv_selected_columns(self):
        """
        Test CSV selecting specific columns.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value,signal_lenght\n2024-01-01,100,300")
            tmp_path = tmp.name

        try:
            df = load_csv(
                file_path=tmp_path,
                column_names=["date", "signal_lenght"],
                date_column=["date"],
                parse_dates=False,
            )

            assert not pd.api.types.is_datetime64_any_dtype(df["date"])
        finally:
            os.remove(tmp_path)

    def test_load_csv_invalid_path(self):
        """
        Test CSV loading with invalid path should raise FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            load_csv(
                file_path="non_existent_file.csv",
                column_names=["date", "value"],
                date_column=["date"],
                parse_dates=True,
            )

    def test_load_csv_invalid_column_names_type(self):
        """
        Test CSV loading with invalid type for column_names.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value\n2024-01-01,100")
            tmp_path = tmp.name

        try:
            with pytest.raises(
                ValueError, match="`column_names` must be a list of strings."
            ):
                load_csv(
                    file_path=tmp_path,
                    column_names="invalid_type",
                    date_column=["date"],
                    parse_dates=True,
                )
        finally:
            os.remove(tmp_path)

    def test_load_csv_invalid_date_column_type(self):
        """
        Test CSV loading with invalid type for date_column.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value\n2024-01-01,100")
            tmp_path = tmp.name

        try:
            with pytest.raises(
                ValueError, match="`date_column` must be a list of strings."
            ):
                load_csv(
                    file_path=tmp_path,
                    column_names=["date", "value"],
                    date_column="date",
                    parse_dates=True,
                )
        finally:
            os.remove(tmp_path)

    def test_load_csv_invalid_parse_dates_type(self):
        """
        Test CSV loading with invalid type for parse_dates.
        """
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write("date,value\n2024-01-01,100")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="`parse_dates` must be a boolean."):
                load_csv(
                    file_path=tmp_path,
                    column_names=["date", "value"],
                    date_column=["date"],
                    parse_dates="yes",
                )
        finally:
            os.remove(tmp_path)

    def test_load_csv_forces_runtime_error(self):
        """
        Force a read_csv failure to test RuntimeError handling.
        """
        with NamedTemporaryFile(mode="w+b", delete=False, suffix=".csv") as tmp:
            tmp.write(b"col1\x00col2\n1,2")
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="Failed to load CSV"):
                load_csv(
                    file_path=tmp_path,
                    column_names=["col1", "col2"],
                    date_column=[],
                    parse_dates=False,
                )
        finally:
            os.remove(tmp_path)
