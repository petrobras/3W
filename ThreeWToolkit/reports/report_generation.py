import matplotlib.pyplot as plt
import pandas as pd
import jinja2

from pathlib import Path
from typing import Any, Dict, Union, Callable

from pylatex import Document, Section, Command, Center, Itemize
from pylatex.utils import NoEscape
from pylatex.package import Package

from ThreeWToolkit.data_visualization import DataVisualization

from ..constants import FIGURES_DIR, LATEX_DIR, REPORTS_DIR, HTML_TEMPLATES_DIR
from ..utils.template_manager import copy_html_support_files, copy_latex_support_files


class ReportGeneration:
    """
    A class for generating and exporting model evaluation reports.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.Series,
        y_train: pd.Series,
        X_test: pd.Series,
        y_test: pd.Series,
        predictions: pd.Series,
        calculated_metrics: dict,
        plot_config: dict,
        title: str,
        author: str = "3W Toolkit Report",
        latex_dir: Path = LATEX_DIR,
        reports_dir: Path = REPORTS_DIR,
        export_report_after_generate: bool = False,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = calculated_metrics
        self.predictions = predictions
        self.calculated_metrics = calculated_metrics
        self.plot_config = plot_config
        self.title = title
        self.author = author
        self.latex_dir = latex_dir

        report_folder = f"report-{self.title}"
        self.reports_dir = reports_dir / report_folder
        self.save_report_after_generate = export_report_after_generate

        self.valid_plots: Dict[str, Callable] = {
            "PlotSeries": DataVisualization.plot_series,
            "PlotMultipleSeries": DataVisualization.plot_multiple_series,
            "PlotCorrelationHeatmap": DataVisualization.correlation_heatmap,
            "PlotFFT": DataVisualization.plot_fft,
            "SeasonalDecompose": DataVisualization.seasonal_decompose,
            "PlotWaveletSpectrogram": DataVisualization.plot_wavelet_spectrogram,
        }

    def _format_metric_name(self, method_name: str) -> str:
        if method_name == "f1":
            return "F1 Score"
        if method_name == "get_roc_auc":
            return "ROC AUC"
        return method_name.replace("get_", "").replace("_", " ").strip().title()

    def _generate_summary_report_latex(self) -> Document:
        """Generates a Beamer presentation summary report as a PyLaTeX Document."""
        print(f"Generating Beamer report: '{self.title}'...")

        doc = Document(documentclass="beamer", document_options=["t,compress"])

        # Pacotes e preâmbulo
        doc.packages.update(
            [
                Package("inputenc", options="utf8"),
                Package("fontenc", options="T1"),
                Package("lmodern"),
                Package("graphicx"),
                Package("booktabs"),
            ]
        )
        doc.preamble.append(Command("usetheme", "petro"))
        doc.preamble.extend(
            [
                Command("title", self.title),
                Command("author", self.author),
                Command("date", NoEscape(r"\today")),
            ]
        )

        doc.append(NoEscape(r"\titlebackground*{assets/background_petro}"))
        doc.append(NoEscape(r"\maketitle"))

        # Slide 1: Metrics
        with doc.create(Section("Performance Evaluation")):
            doc.append(NoEscape(r"\begin{frame}{Figures of Merit}"))
            with doc.create(Center()):
                doc.append(NoEscape(r"\begin{tabular}{l r}\toprule"))
                doc.append(NoEscape(r"\textbf{Metric} & \textbf{Value} \\ \midrule"))
                for name, value in self.calculated_metrics.items():
                    doc.append(
                        NoEscape(f"{name.replace('_', ' ').title()} & {value:.2f} \\\\")
                    )
                doc.append(NoEscape(r"\bottomrule \end{tabular}"))
            doc.append(NoEscape(r"\end{frame}"))

        # Slide 2: Model Info
        with doc.create(Section("Model Overview")):
            doc.append(NoEscape(r"\begin{frame}{Model and Data}"))
            doc.append(NoEscape(r"\begin{block}{Model configuration}"))
            with doc.create(Itemize()) as itemize:
                itemize.add_item(NoEscape(f"Type: {type(self.model).__name__} \\\\"))
                itemize.add_item(NoEscape("Parameters:"))
                with doc.create(Itemize()) as param_itemize:
                    for param, value in self.model.config:
                        param_str = param.replace("_", " ").title()
                        if isinstance(value, (int, float)):
                            # Format simple parameters
                            param_itemize.add_item(NoEscape(f"{param_str}: {value}"))
                        elif isinstance(value, list):
                            # Format lists with commas
                            param_itemize.add_item(
                                NoEscape(f"{param_str}: {', '.join(map(str, value))}")
                            )
                        elif isinstance(value, dict):
                            # Format dictionaries with key-value pairs
                            param_itemize.add_item(
                                NoEscape(
                                    f"{param_str}: {', '.join(f'{k}: {v}' for k, v in value.items())}"
                                )
                            )
                        else:
                            # Handle other types as string representation
                            value_str = str(value).replace("_", " ")
                            if len(value_str) > 50:  # Truncate long strings
                                value_str = value_str[:50] + "..."
                            param_itemize.add_item(
                                NoEscape(f"{param_str}: {value_str}")
                            )

            doc.create(NoEscape(r"\begin{block}{Data Split}"))
            with doc.create(Itemize()) as itemize:
                itemize.add_item(
                    NoEscape(
                        f"Training Samples: {len(self.X_train)} \\\\ Test Samples: {len(self.X_test)}"
                    )
                )
            doc.append(NoEscape(r"\end{block}"))
            doc.append(NoEscape(r"\end{frame}"))

        # Slides 3+: Visualizations
        with doc.create(Section("Visualizations")):
            plot_data = self.get_visualization("latex")
            for plot, details in plot_data.items():
                if plot == "PlotCorrelationHeatmap":
                    width = 0.45
                else:
                    width = 0.7

                img_path = details["img_path"]
                title = details["title"]
                alt = details["alt"]

                doc.append(NoEscape(r"\begin{frame}{" + title + "}"))
                doc.append(NoEscape(r"\begin{figure}\centering"))
                doc.append(
                    NoEscape(
                        f"\\includegraphics[width={width}\\textwidth]{{{str(img_path)}}}"
                    )
                )
                doc.append(NoEscape(f"\\caption{{{alt}}}"))
                doc.append(NoEscape(r"\end{figure}"))
                doc.append(NoEscape(r"\end{frame}"))

        print("Beamer document generated successfully.")
        if self.save_report_after_generate:
            self._save_report_latex(doc=doc, filename=self.title)

        return doc

    def get_visualization(self, format: str):
        """Generates and saves plots based on the provided plot configuration.

        Args:
            format (str): The format of the report, either 'latex' or 'html'.

        Returns:
            dict: A dictionary with plot metadata for inclusion in reports.
        """

        if format not in ["latex", "html"]:
            raise ValueError("Format must be either 'latex' or 'html'.")

        plots_dir = self.reports_dir / format / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)


        plot_paths = {}

        self._check_plot_config(self.plot_config)

        for plot, params in self.plot_config.items():
            plot_func = self.valid_plots.get(plot)
            if not plot_func:
                raise ValueError(f"Plot function for '{plot}' not found.")
            fig, _ = plot_func(**params)

            # figures will be saved with report files to ease inclusion in LaTeX/HTML and external compiling tools
            img_path = plots_dir / f"{self.title}_{plot}.png"
            fig.savefig(img_path, bbox_inches="tight")
            plt.close(fig)

            img_path = Path(img_path)
            img_path = img_path.relative_to(
                self.reports_dir / format
            )  # ensure relative paths for portability

            plot_paths[plot] = {
                "title": params.get("title", "Time Series Plot"),
                "alt": params.get("title", "Time Series Plot"),
                "img_path": str(img_path),
            }

        return plot_paths

    def _check_plot_config(self, plot_config: dict) -> None:
        """Validates the plot configuration dictionary."""

        for plot_name, params in plot_config.items():
            if plot_name not in self.valid_plots:
                raise ValueError(
                    f"Invalid plot name '{plot_name}'. Valid options are: {list(self.valid_plots.keys())}"
                )

            # Basic validation for required parameters
            if plot_name == "PlotSeries":
                if "series" not in params:
                    raise ValueError("Missing 'series' parameter for PlotSeries.")
            elif plot_name == "PlotMultipleSeries":
                if "series_list" not in params or not isinstance(
                    params["series_list"], list
                ):
                    raise ValueError(
                        "Missing or invalid 'series_list' parameter for PlotMultipleSeries."
                    )
            elif plot_name == "PlotCorrelationHeatmap":
                if "df_of_series" not in params or not isinstance(
                    params["df_of_series"], pd.DataFrame
                ):
                    raise ValueError(
                        "Missing or invalid 'df_of_series' parameter for PlotCorrelationHeatmap."
                    )

    def _save_report_latex(self, doc: Document, filename: str) -> None:
        """Compiles and saves a PyLaTeX Document to a PDF file using lualatex."""

        report_path = self.reports_dir / "latex"
        print(f"Saving report to '{report_path}' folder'...")

        report_path.mkdir(parents=True, exist_ok=True)

        doc.generate_tex(filepath=str(report_path / filename))

        print(f"Report saved successfully to '{filename}.tex'")

        copy_latex_support_files(self.latex_dir, report_path)

    def _generate_summary_report_html(
        self, template_name: str = "report_template.html"
    ) -> str:
        """
        Generates a model evaluation report in Markdown format using a Jinja2 template.

        Args:
            template_name (str): The name of the template file located in the 'templates' directory.

        Returns:
            str: A string containing the complete report in Markdown format.
        """
        print(f"Generating Markdown report from template: '{template_name}'...")

        # Set up Jinja2 environment
        # This assumes your templates are in a 'templates' subdirectory.
        # This pathing is robust and should work in most project structures.
        try:
            # Assumes the script is run as part of a module
            template_dir = Path(HTML_TEMPLATES_DIR)
            if not template_dir.exists():
                template_dir = Path.cwd() / "templates"

            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(template_dir),
                autoescape=jinja2.select_autoescape(
                    ["html", "xml"]
                ),  # Good practice for security
            )
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            raise FileNotFoundError(
                f"Template '{template_name}' not found. "
                f"Ensure a 'templates' directory exists at '{template_dir}' "
                "and contains the template file."
            )

        plot_data = self.get_visualization("html")

        context = {
            "title": self.title,
            "author": self.author,
            "generation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "calculated_metrics": self.calculated_metrics,
            "model_type": type(self.model).__name__,
            "model_config": self.model.config.__dict__,
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "plot_data": plot_data,
        }

        # --- 4. Render the template with the data ---
        markdown_output = template.render(context)
        print("Markdown report generated successfully.")

        if self.save_report_after_generate:
            self._save_report_html(
                report_content=markdown_output, filename=f"report-{self.title}.html"
            )

        return markdown_output

    def _save_report_html(self, report_content: str, filename: str) -> None:
        """
        Saves the HTML report content to a file.

        Args:
            report_content (str): The HTML string to be saved.
            filename (str): The name of the file to save (e.g., 'report.md').
        """
        report_path = self.reports_dir / "html"
        report_path.mkdir(parents=True, exist_ok=True)

        file_path = report_path / filename
        print(f"Saving HTML report to '{file_path}'...")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"HTML report saved successfully to '{file_path}'")
        except IOError as e:
            print(f"Error saving HTML file: {e}")

        copy_html_support_files(HTML_TEMPLATES_DIR, FIGURES_DIR, report_path)

    def export_results_to_csv(
        self, results: Dict[str, Any], filename: str
    ) -> pd.DataFrame:
        """Exports results to CSV file."""
        print(f"Exporting results to '{filename}'...")

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        required_keys = [
            "X_test",
            "true_values",
            "predictions",
            "model_name",
            "metrics",
        ]
        if not all(key in results for key in required_keys):
            raise ValueError(
                f"The 'results' dictionary must contain all keys: {required_keys}"
            )

        # Handle X_test which can be a 2D array (matrix)
        if isinstance(results["X_test"], pd.DataFrame):
            df_features = results["X_test"].copy()
        else:
            # Assuming it's a NumPy array
            X_test_arr = results["X_test"]
            if X_test_arr.ndim == 1:
                X_test_arr = X_test_arr.reshape(-1, 1)

            num_features = X_test_arr.shape[1]
            df_features = pd.DataFrame(
                X_test_arr,
                columns=[f"feature_{i + 1}" for i in range(num_features)],
            )

        df_export = df_features.copy()
        df_export["true_values"] = results["true_values"]
        df_export["predictions"] = results["predictions"]
        df_export["model_name"] = results["model_name"]

        for metric_name, metric_value in results["metrics"].items():
            df_export[metric_name] = metric_value

        df_export.to_csv(self.reports_dir / filename, index=True)
        print(f"Successfully exported results to '{self.reports_dir / filename}'.")
        return df_export

    def generate_summary_report(
        self, format, template_name: str = "report_template.html"
    ) -> Union[Document, str]:
        """
        Generates a model evaluation report in either LaTeX (PDF) or HTML format.

        Args:
            format (str): The format of the report to generate, either 'latex' or 'html'
            template_name (str): The name of the template file for HTML reports.
        Returns:
            Union[Document, str]: A PyLaTeX Document for LaTeX reports or a string for HTML reports.
        """
        if self.save_report_after_generate:
            print(f"Reports will be saved to directory: '{self.reports_dir}'")

        if format == "latex":
            return self._generate_summary_report_latex()
        elif format == "html":
            return self._generate_summary_report_html(template_name=template_name)
        else:
            raise ValueError("Format must be either 'latex' or 'html'.")

    def save_report(self, doc: Document | str, filename: str, format: str) -> None:
        """Saves the report in both LaTeX (PDF) and HTML formats.

        Args:
            doc (Document | str): The report content to save. Can be a `Document` object or a raw string containing the report text.
            filename (str): The base name (without extension) of the output file.
            format (str): The output format to use, such as "pdf" or "html".
        """
        if format == "latex":
            self._save_report_latex(doc, filename)
            print("LaTeX report saved successfully")
        elif format == "html":
            self._save_report_html(doc, f"{filename}.html")
            # Convert Markdown to HTML and then to PDF using WeasyPrint
            print("HTML report saved successfully")
        else:
            raise ValueError("Format must be either 'latex' or 'html'.")
