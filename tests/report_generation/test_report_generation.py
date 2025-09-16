import os
import pytest
import pandas as pd
from unittest.mock import MagicMock

from ThreeWToolkit.reports.report_generation import ReportGeneration
from .mocks import mock_model, mock_time_series_data        
from pathlib import Path

from ThreeWToolkit.constants import REPORTS_DIR, LATEX_DIR

@pytest.mark.skip(reason="This test class was disabled for LaTeX compiler validation.")
class TestReportGeneration:
    """Tests for the ReportGeneration class."""

    @pytest.mark.parametrize("input_name, expected_output", [
        ("get_neg_mean_absolute_error", "Neg Mean Absolute Error"),
        ("get_f1", "F1 Score"),
        ("get_roc_auc", "ROC AUC"),
        ("get_explained_variance", "Explained Variance"),
    ])
    def test_format_metric_name(self, input_name, expected_output):
        """Test the static helper for formatting metric names."""
        report_generation = ReportGeneration()

        assert report_generation._format_metric_name(input_name) == expected_output

    def test_generate_summary_report(self, mocker, mock_model, mock_time_series_data):
        """
        Test the main report generation logic. Mocks all external calls (plotting).
        Verifies that the generated LaTeX document contains the correct information.
        """
        # Mock all DataVisualization methods to avoid creating plots
        mocker.patch('ThreeWToolkit.reports.report_generation.DataVisualization.plot_multiple_series', return_value="mock_path_pred.png")
        mocker.patch('ThreeWToolkit.reports.report_generation.DataVisualization.plot_fft', return_value="mock_path_fft.png")
        mocker.patch('ThreeWToolkit.reports.report_generation.DataVisualization.seasonal_decompose', return_value="mock_path_decomp.png")
        mocker.patch('ThreeWToolkit.reports.report_generation.DataVisualization.correlation_heatmap', return_value="mock_path_heatmap.png")
        mocker.patch('ThreeWToolkit.reports.report_generation.DataVisualization.plot_wavelet_spectrogram', return_value="mock_path_wavelet.png")

        metrics_to_include = [
            "get_accuracy",
            "get_f1",
            "get_roc_auc",
            "get_explained_variance"
        ]

        report_generation = ReportGeneration()

        doc = report_generation.generate_summary_report(
            model=mock_model,
            metrics=metrics_to_include,
            title="My Test Report",
            **mock_time_series_data
        )

        # Get the LaTeX source as a string
        latex_source = doc.dumps()

        assert r"\title{My Test Report}" in latex_source
        assert r"Type: MockModel" in latex_source
        assert r"Method: rolling threshold" in latex_source # Check parameter parsing
        assert r"Window: 3" in latex_source # Check list parameter parsing
        assert r"Threshold: 1.5" in latex_source # Check list parameter parsing
        assert r"Training Samples: 40" in latex_source
        assert r"Test Samples: 10" in latex_source
        
        assert r"F1 Score" in latex_source
        assert r"ROC AUC" in latex_source
        assert r"Accuracy" in latex_source
        assert r"Explained Variance" in latex_source

        assert r"\includegraphics[width=0.9\textwidth]{mock_path_pred.png}" in latex_source
        assert r"\includegraphics[width=0.8\textwidth]{mock_path_fft.png}" in latex_source
        assert r"\includegraphics[width=0.45\textwidth]{mock_path_decomp.png}" in latex_source

    def test_save_report(self, mocker, tmp_path):
        """
        Test the save_report method against the new implementation, mocking dependencies
        to ensure proper isolation.
        """
        test_filename = "my_test_report"
        
        mocker.patch('ThreeWToolkit.reports.report_generation.REPORTS_DIR', tmp_path)
        mocker.patch('ThreeWToolkit.reports.report_generation.LATEX_DIR', tmp_path)

        mock_manager = mocker.patch('ThreeWToolkit.reports.report_generation.latex_environment', MagicMock())

        mock_doc = MagicMock()
        report_generation = ReportGeneration()

        report_generation.save_report(mock_doc, test_filename)

        mock_manager.assert_called_once_with(tmp_path)

        expected_report_path = tmp_path / f"report-{test_filename}"
        mock_doc.generate_pdf.assert_called_once_with(
            test_filename,
            clean=True,
            clean_tex=True,
            compiler="lualatex",
            compiler_args=[f"--output-directory={expected_report_path}"],
            silent=False,
        )

    def test_export_results_to_csv_success(self, tmp_path):
        """
        Verify that export_results_to_csv creates a valid DataFrame and saves it correctly.
        Uses the pytest `tmp_path` fixture to handle file creation cleanly.
        """
        output_dir = tmp_path / "exports"
        output_dir.mkdir()
        filename = output_dir / "test_results.csv"

        results_data = {
            'X_test': pd.Series([1, 2, 3], name="x_test_data"),
            'y_test': pd.Series([1.1, 2.2, 3.3], name="y_test_data"),
            'prediction': pd.Series([1.0, 2.1, 3.2], name="preds"),
            'model_name': 'MyMockModel',
            'metrics': {
                'mae': 0.1,
                'rmse': 0.15
            }
        }

        report_generation = ReportGeneration()

        returned_df = report_generation.export_results_to_csv(results_data, str(filename))

        assert isinstance(returned_df, pd.DataFrame)
        expected_columns = ['X_test', 'y_test', 'prediction', 'model_name', 'mae', 'rmse']
        assert all(col in returned_df.columns for col in expected_columns)
        assert returned_df['model_name'].iloc[0] == 'MyMockModel'
        assert returned_df['mae'].iloc[0] == 0.1

        assert filename.exists()

        df_from_csv = pd.read_csv(filename, index_col=0)
        assert df_from_csv.shape == (3, 6)
        assert df_from_csv['prediction'].iloc[1] == 2.1

    def test_export_results_to_csv_raises_error_on_missing_key(self):
        """
        Verify that the function raises a ValueError if the results dictionary
        is missing one or more of the required keys.
        """
        # Setup: Create results data that is missing the 'model_name' key
        incomplete_results = {
            'X_test': pd.Series([1, 2, 3]),
            'y_test': pd.Series([1.1, 2.2, 3.3]),
            'prediction': pd.Series([1.0, 2.1, 3.2]),
            # 'model_name': 'MyMockModel', # Intentionally missing
            'metrics': {'mae': 0.1}
        }
        report_generation = ReportGeneration()

        with pytest.raises(ValueError) as excinfo:
            report_generation.export_results_to_csv(incomplete_results, "dummy_filename.csv")

        assert "must contain all keys" in str(excinfo.value)