import matplotlib.pyplot as plt
import pandas as pd
import jinja2

from pathlib import Path
from typing import Any, Dict, Union, Callable

from pylatex import Document, Section, Command, Center, Itemize
from pylatex.utils import NoEscape
from pylatex.package import Package

from ThreeWToolkit.data_visualization import DataVisualization

from ..constants import LATEX_DIR, REPORTS_DIR, HTML_TEMPLATES_DIR, HTML_ASSETS_DIR
from ..utils.template_manager import copy_html_support_files, copy_latex_support_files


class ReportGeneration:
    """A class for generating and exporting model evaluation reports.
    This class orchestrates the creation of comprehensive reports for machine
    learning models. It takes training and testing data, model predictions,
    calculated metrics, and plotting configurations to produce reports in
    various formats, including LaTeX (Beamer presentations) and HTML. The class
    can also export the raw prediction results to a CSV file for further analysis.

    Attributes:
        model (Any): The trained machine learning model object.
        X_train (pd.Series): The training feature data.
        y_train (pd.Series): The training target data.
        X_test (pd.Series): The testing feature data.
        y_test (pd.Series): The testing target data.
        predictions (pd.Series): The model's predictions on the test set.
        calculated_metrics (dict): A dictionary of pre-calculated performance metrics.
        plot_config (dict): A dictionary defining the plots to be included in the report.
        title (str): The title of the report.
        author (str): The author of the report.
        latex_dir (Path): The directory to store intermediate LaTeX files.
        reports_dir (Path): The base directory where the final report folder will be created.
        save_report_after_generate (bool): If True, the report is saved immediately after generation.
        valid_plots (Dict[str, Callable]): A mapping of valid plot names to their corresponding functions.

    Methods:
        generate_summary_report(format: str, template_name: str = "report_template.html") -> Union[Document, str]:
            Generates a model evaluation report in either LaTeX (PDF) or HTML format.
        save_report(doc: Document | str, filename: str, format: str) -> None:
            Saves the report in both LaTeX (PDF) and HTML formats.
        export_results_to_csv(results: Dict[str, Any], filename: str) -> pd.DataFrame:
            Exports machine learning model results to a CSV file.
        get_visualization(format: str) -> dict:
            Generates and saves plots based on the provided plot configuration.
    Private Methods:
        _generate_summary_report_latex() -> Document:
            Generates a summary report as a LaTeX Beamer presentation.
        _generate_summary_report_html(template_name: str = "report_template.html") -> str:
            Generates a model evaluation report in Markdown format using a Jinja2 template.
        _save_report_latex(doc: Document, filename: str) -> None:
            Saves a PyLaTeX Document as a .tex file along with its support files.
        _save_report_html(report_content: str, filename: str) -> None:
            Saves the HTML report content to a file and copies necessary assets.
        _check_plot_config(plot_config: dict) -> None:
            Validates the plot configuration dictionary.

    Note:
        This class is designed to be flexible and extensible, allowing users to
        customize the report generation process according to their needs.
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
        plot_config: Union[dict, None],
        title: str,
        author: str = "3W Toolkit Report",
        latex_dir: Path = LATEX_DIR,
        reports_dir: Path = REPORTS_DIR,
        export_report_after_generate: bool = False,
    ):
        """Initializes the ReportGeneration class.

        This constructor sets up the necessary data and configuration for generating
        the 3W model report. It takes datasets, model predictions,
        metrics, and various configuration options as input.

        Args:
            model (Any): The trained machine learning model object.
            X_train (pd.Series): The training feature data.
            y_train (pd.Series): The training target data.
            X_test (pd.Series): The testing feature data.
            y_test (pd.Series): The testing target data.
            predictions (pd.Series): The model's predictions on the test set.
            calculated_metrics (dict): A dictionary of pre-calculated performance metrics.
            plot_config (Union[dict, None]): A dictionary defining the plots to be
                included in the report. If None, an empty dictionary is used.
            title (str): The title of the report.
            author (str, optional): The author of the report.
                Defaults to "3W Toolkit Report".
            latex_dir (Path, optional): The directory to store intermediate LaTeX files.
                Defaults to the constant LATEX_DIR.
            reports_dir (Path, optional): The base directory where the final report
                folder will be created. Defaults to the constant REPORTS_DIR.
            export_report_after_generate (bool, optional): If True, the report is
                saved (e.g., to .html or .tex) immediately after generation. Defaults to False.
        """

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = calculated_metrics
        self.predictions = predictions
        self.calculated_metrics = calculated_metrics

        if plot_config is None:
            plot_config = {}

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

    def _generate_summary_report_latex(self) -> Document:
        """Generates a summary report as a LaTeX Beamer presentation.

        This method constructs a multi-slide Beamer presentation using the PyLaTeX
        library. The presentation is structured into several sections:

        1.  **Title Page**: Displays the report's title, author, and the current date,
            set against a custom background.
        2.  **Performance Evaluation**: A slide titled "Figures of Merit" that presents
            key performance metrics in a tabular format.
        3.  **Model Overview**: A slide titled "Model and Data" which details the
            model's configuration (type and parameters) and the data split
            (number of training and test samples).
        4.  **Visualizations**: A series of slides, one for each visualization
            (e.g., plots, heatmaps) generated by the `get_visualization` method.
            Each slide includes the plot image and a descriptive caption.

        The method uses a custom Petrobras Beamer theme and includes necessary LaTeX
        packages for formatting. If `self.save_report_after_generate` is True,
        the generated document is automatically saved as a .tex file and compiled
        into a PDF.

        Note:
            This is a private method intended for internal use by the report
            generation system. It relies on several instance attributes being
            previously set.

        Returns:
            pylatex.Document: The generated Beamer presentation as a PyLaTeX
                              Document object.
        """
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
        """Validates the plot configuration dictionary.

        This method iterates through the provided plot configuration, ensuring
        that each specified plot is a valid type and that all its required
        parameters are present and correctly formatted.
        Args:
            plot_config (dict): A dictionary where keys are the names of the
                plots to be generated (e.g., "PlotSeries", "PlotCorrelationHeatmap")
                and values are dictionaries containing the parameters for each plot.
        Raises:
            ValueError: If a plot name is not recognized, or if a required
                parameter for a specific plot type is missing or has an
                invalid type (e.g., 'series_list' not being a list).
        """

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
        """Saves a PyLaTeX Document as a .tex file along with its support files.

        This private method handles the process of generating a .tex file from a
        given PyLaTeX `Document` object. It creates a 'latex' subdirectory within
        the main reports directory (`self.reports_dir`) if it doesn't already
        exist. The generated .tex file is saved there. Additionally, it copies
        necessary LaTeX support files (e.g., images, style files) from a source
        directory (`self.latex_dir`) to the output directory to ensure the .tex
        file can be compiled correctly.

        Args:
            doc (Document): The PyLaTeX Document object to be saved.
            filename (str): The base name for the output .tex file (e.g., 'my_report').
                The '.tex' extension will be added automatically.

        Side Effects:
            - Creates a directory at `self.reports_dir / "latex"`.
            - Writes a .tex file to the created directory.
            - Copies support files from `self.latex_dir` to the output directory.
            - Prints status messages to the console.
        """

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
            template_name (str, optional): The name of the Jinja2 template file.
                This file should be located in the directory specified by
                `HTML_TEMPLATES_DIR` or a 'templates' subdirectory relative to the
                current working directory. Defaults to "report_template.html".

        Raises:
            FileNotFoundError: If the specified `template_name` is not found in the
                configured template directory.

        Returns:
            str: A string containing the complete, rendered report in HTML format.
        """
        print(f"Generating HTML report from template: '{template_name}'...")

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
        """Saves the HTML report content to a file and copies necessary assets.

        This private method handles the process of persisting a generated HTML report.
        It first ensures the target directory (`[reports_dir]/html/`) exists,
        creating it if necessary. It then writes the provided HTML content to the
        specified file within that directory. Finally, it copies over any required
        support files (e.g., CSS, JavaScript, images) to the same directory to
        ensure the HTML report can be rendered correctly as a standalone file.

            report_content (str): The complete HTML content of the report as a string.
            filename (str): The name for the output file (e.g., 'analysis_report.html').

        Raises:
            IOError: If there is an error writing the HTML file to the disk.

        Side Effects:
            - Creates the `[reports_dir]/html/` directory if it doesn't exist.
            - Writes a new file to the filesystem at `[reports_dir]/html/[filename]`.
            - Copies support files to the `[reports_dir]/html/` directory.
            - Prints status messages to the console during execution.
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

        copy_html_support_files(HTML_TEMPLATES_DIR, HTML_ASSETS_DIR, report_path)

    def export_results_to_csv(
        self, results: Dict[str, Any], filename: str
    ) -> pd.DataFrame:
        """Exports machine learning model results to a CSV file.

        This method takes a dictionary of results from a model evaluation,
        formats it into a pandas DataFrame, and saves it as a CSV file in the
        designated reports directory (`self.reports_dir`). The directory is
        created if it does not exist.

        The output CSV file will contain the input features (`X_test`), the true
        target values, the model's predictions, the model's name, and the
        calculated performance metrics.

        Args:
            results (Dict[str, Any]): A dictionary containing the model evaluation
                results. It must include the following keys:
                - 'X_test' (pd.DataFrame or np.ndarray): The test features.
                - 'true_values' (array-like): The actual target values.
                - 'predictions' (array-like): The model's predicted values.
                - 'model_name' (str): The name of the model.
                - 'metrics' (Dict[str, float]): A dictionary where keys are metric
                  names (e.g., 'MAE', 'MSE') and values are the corresponding
                  scores.
            filename (str): The name for the output CSV file (e.g., 'results.csv').

        Returns:
            pd.DataFrame: The DataFrame containing the combined results that was
                saved to the CSV file.

        Raises:
            ValueError: If the `results` dictionary does not contain all the
                required keys.
        """
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
