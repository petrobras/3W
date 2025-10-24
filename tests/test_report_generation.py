import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from pylatex import Document


def test_initialization(
    report_generator_instance,
):
    """Test if the class is initialized correctly."""
    assert report_generator_instance.title == "Test_Report"
    assert report_generator_instance.author == "3W Toolkit Report"
    assert "Accuracy" in report_generator_instance.metrics
    assert report_generator_instance.reports_dir.name == "report-Test_Report"


def test_check_plot_config_valid(
    report_generator_instance,
):
    """Test that valid plot configurations pass without error."""
    try:
        valid_config = {
            "PlotSeries": {"series": pd.Series([1, 2])},
            "PlotCorrelationHeatmap": {
                "df_of_series": pd.DataFrame({"a": [1], "b": [2]})
            },
        }
        report_generator_instance._check_plot_config(valid_config)
    except ValueError:
        pytest.fail("Valid plot config raised ValueError unexpectedly.")


def test_check_plot_config_invalid_plot_name(
    report_generator_instance,
):
    """Test that an invalid plot name raises a ValueError."""
    invalid_config = {"InvalidPlotName": {}}
    with pytest.raises(ValueError, match="Invalid plot name 'InvalidPlotName'"):
        report_generator_instance._check_plot_config(invalid_config)


def test_check_plot_config_missing_parameter(
    report_generator_instance,
):
    """Test that a missing required parameter raises a ValueError."""
    invalid_config = {"PlotSeries": {"title": "A plot with no data"}}
    with pytest.raises(ValueError, match="Missing 'series' parameter"):
        report_generator_instance._check_plot_config(invalid_config)


@patch("ThreeWToolkit.reports.report_generation.DataVisualization.plot_multiple_series")
@patch("ThreeWToolkit.reports.report_generation.DataVisualization.plot_series")
@patch("ThreeWToolkit.reports.report_generation.plt")
def test_get_visualization(
    mock_plt, mock_plot_series, mock_plot_multiple, report_generator_instance
):
    """Test that visualization generation calls the correct plotters and saves files."""
    # Setup mock figures that can be saved
    mock_fig = MagicMock()
    mock_plot_series.return_value = mock_fig
    mock_plot_multiple.return_value = mock_fig

    result = report_generator_instance.get_visualization("html")
    result = report_generator_instance.get_visualization("latex")  # Run both formats

    # Assert that the output dictionary has the correct structure
    assert "PlotSeries" in result
    assert result["PlotSeries"]["title"] == "Test Series Plot"
    assert Path(result["PlotSeries"]["img_path"]).name == "Test_Report_PlotSeries.png"

    # Assert that plt.close was called to free memory
    assert (
        mock_plt.close.call_count == 2 * 2
    )  # Called for each plot in both format calls


@pytest.mark.skip(reason="This test class was disabled temporarily.")
@patch("ThreeWToolkit.reports.report_generation.ReportGeneration.get_visualization")
def test_generate_summary_report_latex(mock_get_viz, report_generator_instance):
    """Test LaTeX report generation logic without file I/O."""

    mock_get_viz.return_value = {
        "plot1": {
            "img_path": Path("plots") / "plot1.png",
            "title": "Plot One",
            "alt": "Alt One",
        }
    }

    doc = report_generator_instance._generate_summary_report_latex()

    assert isinstance(doc, Document)

    # Check for key content without being too specific about LaTeX syntax
    latex_str = doc.dumps()
    assert "Test\\_Report" in latex_str
    assert "3W Toolkit Report" in latex_str
    assert "Accuracy & 0.95" in latex_str  # Check if metrics are there
    assert "Parameter Alpha: 0.1" in latex_str  # Check if model params are there
    assert "plots/plot1.png" in latex_str  # Check if image path is included


@pytest.mark.skip(reason="This test class was disabled temporarily.")
@patch("ThreeWToolkit.reports.report_generation.ReportGeneration.get_visualization")
def test_generate_summary_report_html(
    mock_get_viz, report_generator_instance, tmp_path
):
    """Test HTML/Markdown report generation from a Jinja2 template."""
    # Create a dummy template file
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "report_template.html"

    # Point the constant to our temporary directory for the test
    with patch(
        "ThreeWToolkit.reports.report_generation.HTML_TEMPLATES_DIR", template_dir
    ):
        # Mock get_visualization to return predictable data
        mock_get_viz.return_value = {
            "plot1": {
                "img_path": Path("plots") / "plot1.png",
                "title": "Plot One",
                "alt": "Plot Alt Text",
            }
        }

        # The template text has been corrected here (no hyphen in the image tag)
        template_file.write_text(
            """
# {{ title }}
- Metric: {{ calculated_metrics.Accuracy }}
- Model: {{ model_type }}
{% for name, plot in plot_data.items() %}
![{{ plot.alt }}]({{ plot.img_path }})
{% endfor %}
        """
        )

        markdown_output = report_generator_instance._generate_summary_report_html(
            template_name="report_template.html"
        )

    # This assertion will now pass
    assert "# Test_Report" in markdown_output
    assert "- Metric: 0.95" in markdown_output
    assert "- Model: MockModel" in markdown_output
    assert "![Plot Alt Text](plots/plot1.png)" in markdown_output


@patch("ThreeWToolkit.reports.report_generation.copy_html_support_files")
def test_save_report_html(mock_copy_files, report_generator_instance, tmp_path):
    """Test saving a string content to an HTML file."""
    content = "<h1>Test Content</h1>"
    filename = "test_output.html"

    # Override the reports_dir to use the root of tmp_path for simplicity
    report_generator_instance.reports_dir = tmp_path

    report_generator_instance._save_report_html(content, filename)

    mock_copy_files.assert_called_once()

    output_file = tmp_path / "html" / filename
    assert output_file.exists()
    assert output_file.read_text() == content


@patch("ThreeWToolkit.reports.report_generation.copy_latex_support_files")
def test_save_report_latex(mock_copy_files, report_generator_instance, tmp_path):
    """Test that the LaTeX saving logic calls generate_tex correctly."""
    mock_doc = MagicMock(spec=Document)
    report_generator_instance.reports_dir = tmp_path  # Simplify path

    report_generator_instance._save_report_latex(mock_doc, "MyLatexReport")

    # Check that the latex_environment context manager was used
    mock_copy_files.assert_called_once()

    # Check that the file generation was called with the correct path
    expected_path = str(tmp_path / "latex" / "MyLatexReport")
    mock_doc.generate_tex.assert_called_once_with(filepath=expected_path)


def test_generate_summary_report_dispatcher(report_generator_instance):
    """Test the main dispatcher method `generate_summary_report`."""
    with patch.object(
        report_generator_instance, "_generate_summary_report_latex"
    ) as mock_latex:
        report_generator_instance.generate_summary_report(format="latex")
        mock_latex.assert_called_once()

    with patch.object(
        report_generator_instance, "_generate_summary_report_html"
    ) as mock_html:
        report_generator_instance.generate_summary_report(format="html")
        mock_html.assert_called_once()

    with pytest.raises(ValueError, match="Format must be either 'latex' or 'html'"):
        report_generator_instance.generate_summary_report(format="invalid_format")


def test_export_with_dataframe_xtest(report_generator_instance, sample_results_dict):
    """
    Tests the main success path where X_test is a DataFrame.
    Verifies file creation, returned DataFrame content, and CSV content.
    """
    filename = "test_report.csv"

    # --- Action ---
    returned_df = report_generator_instance.export_results_to_csv(
        results=sample_results_dict, filename=filename
    )

    # --- Verification ---
    # 1. Check if the file was created in the correct temporary directory
    output_path = report_generator_instance.reports_dir / filename
    assert output_path.exists()

    # 2. Read the created CSV and verify its contents
    df_from_csv = pd.read_csv(
        output_path, index_col=0
    )  # index_col=0 because to_csv saves the index

    # 3. Check columns in both the returned DF and the one from the CSV
    expected_cols = [
        "feature_A",
        "feature_B",
        "true_values",
        "predictions",
        "model_name",
        "Accuracy",
        "F1 Score",
    ]
    assert returned_df.columns.tolist() == expected_cols
    assert df_from_csv.columns.tolist() == expected_cols

    # 4. Check data integrity
    assert df_from_csv["model_name"].iloc[0] == "MyAwesomeModel"
    assert df_from_csv["Accuracy"].iloc[0] == 0.6
    assert df_from_csv["predictions"].tolist() == [1, 1, 1, 0, 0]


def test_export_with_numpy_xtest(report_generator_instance, sample_results_dict):
    """Tests the logic path where X_test is a 1D and 2D NumPy array."""
    # Checking 1D NumPy array
    sample_results_dict["X_test"] = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    df_1d = report_generator_instance.export_results_to_csv(
        sample_results_dict, "report_1d.csv"
    )

    # Check that it was correctly converted to a column named 'feature_1'
    assert "feature_1" in df_1d.columns
    assert df_1d["feature_1"].tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]

    # Checking 2D NumPy array
    sample_results_dict["X_test"] = np.array(
        [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
    )
    df_2d = report_generator_instance.export_results_to_csv(
        sample_results_dict, "report_2d.csv"
    )

    # Check that it was correctly converted to 'feature_1', 'feature_2', etc.
    assert "feature_1" in df_2d.columns
    assert "feature_2" in df_2d.columns
    assert df_2d["feature_2"].tolist() == [10, 20, 30, 40, 50]


def test_export_missing_keys_raises_error(
    report_generator_instance, sample_results_dict
):
    """Tests that a ValueError is raised if the results dict is missing keys."""
    # Remove a required key
    del sample_results_dict["model_name"]

    # Use pytest.raises to assert that a specific exception is thrown
    with pytest.raises(ValueError, match="must contain all keys"):
        report_generator_instance.export_results_to_csv(
            results=sample_results_dict, filename="should_not_be_created.csv"
        )
