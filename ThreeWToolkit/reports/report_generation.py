import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Any, List, Dict

from statsmodels.tsa.seasonal import seasonal_decompose
from pylatex import Document, Section, Command, Center, Itemize
from pylatex.utils import NoEscape
from pylatex.package import Package

from ..constants import PLOTS_DIR, LATEX_DIR, REPORTS_DIR
from ..utils.latex_manager import latex_environment
from ..metrics import accuracy_score, f1_score, roc_auc_score, explained_variance_score


class DataVisualization:
    @staticmethod
    def _save_plot(title: str) -> str:
        """Helper to save a plot to the 'plots' directory."""
        plot_dir = Path(PLOTS_DIR)

        # Create the directory if it doesn't exist.
        plot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{title.replace(' ', ' ').lower()}.png"

        filepath = plot_dir / filename

        plt.savefig(filepath)
        plt.close()

        print(f"DataVisualization: Chart saved to '{filepath}'")
        return str(filepath)

    @staticmethod
    def plot_multiple_series(
        series_list: List[pd.Series],
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> str:
        for i, series in enumerate(series_list):
            if isinstance(series, np.ndarray):
                series_list[i] = pd.Series(series)
            elif not isinstance(series, pd.Series):
                raise ValueError("Input series must be pandas Series or numpy ndarray.")

        plt.figure(figsize=(10, 5))
        for series, label in zip(series_list, labels):
            print(series)
            plt.plot(series.index, series.values, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    @staticmethod
    def plot_fft(series: pd.Series, title: str = "FFT Analysis") -> str:
        """Calculates and plots the Fast Fourier Transform of a series."""
        N = len(series)
        T = 1.0 / N  # Sample spacing
        yf = np.fft.fft(series.values)
        xf = np.fft.fftfreq(N, T)[: N // 2]

        plt.figure(figsize=(10, 5))
        plt.plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
        plt.grid()
        plt.title(title)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    @staticmethod
    def seasonal_decompose(
        series: pd.Series, model: str = "additive", period: int = 12
    ) -> str:
        """Performs and plots seasonal decomposition."""
        result = seasonal_decompose(series, model=model, period=period)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        result.observed.plot(ax=ax1, legend=False)
        ax1.set_ylabel("Observed")
        result.trend.plot(ax=ax2, legend=False)
        ax2.set_ylabel("Trend")
        result.seasonal.plot(ax=ax3, legend=False)
        ax3.set_ylabel("Seasonal")
        result.resid.plot(ax=ax4, legend=False)
        ax4.set_ylabel("Residual")
        plt.suptitle("Seasonal Decomposition", y=0.94)
        plt.tight_layout()
        return DataVisualization._save_plot("Seasonal_Decomposition")

    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame, title: str = "Correlation Heatmap"
    ) -> str:
        """Plots a correlation heatmap for a DataFrame."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_of_series.corr(), annot=True, cmap="viridis", fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        return DataVisualization._save_plot(title)

    # --- Other mock implementations from UML ---
    @staticmethod
    def plot_series(series: pd.Series, title: str, xlabel: str, ylabel: str) -> str:
        return DataVisualization.plot_multiple_series(
            [series], [series.name], title, xlabel, ylabel
        )

    @staticmethod
    def plot_wavelet_spectrogram(
        series: pd.Series, title: str = "Wavelet Spectrogram"
    ) -> str:
        """Mock plot for a wavelet spectrogram."""
        plt.figure(figsize=(10, 5))
        # In a real scenario, use libraries like pywt. For now, a mock image.
        mock_spectrogram = np.random.rand(50, len(series))
        plt.imshow(mock_spectrogram, aspect="auto", cmap="inferno")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Frequency Scale")
        return DataVisualization._save_plot(title)


class ReportGeneration:
    """
    A class for generating and exporting model evaluation reports.

    This class provides methods to create comprehensive PDF summary reports
    as Beamer presentations and to export numerical results to CSV files.
    """

    def _format_metric_name(self, method_name: str) -> str:
        """Formats a metric's method name into a human-readable title.

        Handles special cases like 'get_f1' and general cases by removing
        the 'get_' prefix and converting snake_case to Title Case.

        Args:
            method_name (str): The raw method name from the Metrics class
                               (e.g., 'get_neg_mean_absolute_error').

        Returns:
            str: A formatted, human-readable string for use in reports
                 (e.g., "Negative Mean Absolute Error").
        """

        if method_name == "get_f1":
            return "F1 Score"
        if method_name == "get_roc_auc":
            return "ROC AUC"
        return method_name.replace("get_", "").replace("_", " ").strip().title()

    def generate_summary_report(
        self,
        model: Any,
        X_train: pd.Series,
        y_train: pd.Series,
        X_test: pd.Series,
        y_test: pd.Series,
        metrics: List[str],
        title: str,
    ) -> Document:
        """Generates a Beamer presentation summary report as a PyLaTeX Document.

        This method orchestrates the creation of a multi-slide PDF report
        containing performance metrics, model configuration details, and a
        series of data visualizations.

        Args:
            model (Any): The trained model object. Must implement a `get_params()`
                         method.
            X_train (pd.Series): The training data features.
            y_train (pd.Series): The training data target values.
            X_test (pd.Series): The test data features.
            y_test (pd.Series): The test data target values.
            metrics (List[str]): A list of metric method names from the Metrics
                                 class (e.g., 'get_explained_variance') to be
                                 calculated and displayed in the report.
            title (str): The main title for the presentation.

        Returns:
            Document: A PyLaTeX `Document` object containing the complete Beamer
                      presentation, ready to be compiled and saved.
        """

        print(f"Generating Beamer report: '{title}'...")
        # Map the metric names from the config to the actual metric functions.
        metric_function_map = {
            "get_accuracy": accuracy_score,
            "get_f1": f1_score,
            "get_roc_auc": roc_auc_score,
            "get_explained_variance": explained_variance_score,
        }

        # Calculate predictions once to avoid re-computing for each metric.
        y_pred = model.predict(X_test)
        calculated_metrics = []

        for name in metrics:
            if name in metric_function_map:
                metric_func = metric_function_map[name]
                value = metric_func(y_true=y_test, y_pred=y_pred)
                calculated_metrics.append(
                    (self._format_metric_name(name), f"{value:.3f}")
                )

        doc = Document(documentclass="beamer", document_options=["t,compress"])

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
                Command("title", title),
                Command("author", "3W Toolkit Report"),
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
                for name, value in calculated_metrics:
                    doc.append(NoEscape(f"{name} & {value} \\\\"))
                doc.append(NoEscape(r"\bottomrule \end{tabular}"))
            doc.append(NoEscape(r"\end{frame}"))

        # Slide 2: Model Info
        with doc.create(Section("Model Overview")):
            doc.append(NoEscape(r"\begin{frame}{Model and Data}"))
            doc.append(NoEscape(r"\begin{block}{Model configuration}"))
            with doc.create(Itemize()) as itemize:
                itemize.add_item(NoEscape(f"Type: {type(model).__name__} \\\\"))
                itemize.add_item(NoEscape("Parameters:"))
                with doc.create(Itemize()) as param_itemize:
                    for param, value in model.get_params().items():
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
                        f"Training Samples: {len(X_train)} \\\\ Test Samples: {len(X_test)}"
                    )
                )
            doc.append(NoEscape(r"\end{block}"))
            doc.append(NoEscape(r"\end{frame}"))

        # Slides 3+: Visualizations
        with doc.create(Section("Visualizations")):
            # Plot 1: Predicted vs Actual
            doc.append(NoEscape(r"\begin{frame}{Visualization: Predicted vs. Actual}"))
            img_path = DataVisualization.plot_multiple_series(
                [y_test, model.predict(X_test)],
                ["Actual Signal", "Predicted Signal"],
                "Signal_Prediction",
                "Time Step",
                "Amplitude",
            )
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(f"\\includegraphics[width=0.9\\textwidth]{{{img_path}}}")
            )
            doc.append(NoEscape(r"\caption{Comparison of Actual vs Predicted Signals}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

            # Plot 2: FFT Analysis
            doc.append(NoEscape(r"\begin{frame}{Signal Analysis: FFT}"))
            fft_img_path = DataVisualization.plot_fft(y_test)
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(f"\\includegraphics[width=0.8\\textwidth]{{{fft_img_path}}}")
            )
            doc.append(NoEscape(r"\caption{FFT Analysis of the Signal}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

            # Plot 3: Seasonal Decomposition
            doc.append(NoEscape(r"\begin{frame}{Signal Analysis: Decomposition}"))
            decomp_img_path = DataVisualization.seasonal_decompose(
                y_test, period=20
            )  # Period matching one of the sine waves
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(
                    f"\\includegraphics[width=0.45\\textwidth]{{{decomp_img_path}}}"
                )
            )
            doc.append(NoEscape(r"\caption{Seasonal Decomposition of the Signal}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

            # Plot 4: Correlation Heatmap
            doc.append(NoEscape(r"\begin{frame}{Correlation Heatmap}"))
            heatmap_df = pd.DataFrame(
                {
                    "Actual": y_test,
                    "Predicted": model.predict(X_test),
                    "Residuals": y_test - model.predict(X_test),
                }
            )
            heatmap_img_path = DataVisualization.correlation_heatmap(heatmap_df)
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(
                    f"\\includegraphics[width=0.5\\textwidth]{{{heatmap_img_path}}}"
                )
            )
            doc.append(NoEscape(r"\caption{Correlation Heatmap}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

            # Plot 5: Wavelet Spectrogram (Mock)
            doc.append(NoEscape(r"\begin{frame}{Wavelet Spectrogram}"))
            wavelet_img_path = DataVisualization.plot_wavelet_spectrogram(y_test)
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(
                    f"\\includegraphics[width=0.8\\textwidth]{{{wavelet_img_path}}}"
                )
            )
            doc.append(NoEscape(r"\caption{Wavelet Spectrogram of the Signal}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

        # Finalize document
        print("Beamer document generated successfully.")
        return doc

    def save_report(self, doc: Document, filename: str) -> None:
        """Compiles and saves a PyLaTeX Document to a PDF file using lualatex.

        This method configures the environment for the LaTeX compiler by setting
        the TEXINPUTS variable, allowing it to find local theme files and
        assets. It saves the final PDF into a dedicated subdirectory and
        restores the environment upon completion.

        Args:
            doc (Document): The PyLaTeX `Document` object to be compiled.
            filename (str): The base name for the output PDF and report directory
                          (e.g., 'my_report').
        """
        # Localization for LaTeX files (template, assets, etc)
        with latex_environment(LATEX_DIR):
            # Generate the PDF
            report_folder = f"report-{filename}"
            report_path = REPORTS_DIR / report_folder
            print(f"Saving report to '{report_path}' folder'...")

            report_path.mkdir(parents=True, exist_ok=True)

            doc.generate_pdf(
                filename,
                clean=True,
                clean_tex=True,
                compiler="lualatex",
                compiler_args=[f"--output-directory={report_path}"],
                silent=False,
            )
            print(f"Report saved successfully to '{filename}.pdf'")

    def export_results_to_csv(
        self, results: Dict[str, Any], filename: str
    ) -> pd.DataFrame:
        """Exports a comprehensive dictionary of experiment results to a CSV file.

        This method creates a single DataFrame where each row corresponds to a
        data point in the test set. It includes columns for the test data,
        predictions, model name, and all calculated metrics. The metadata
        (model name, metrics) is repeated on each row to create
        self-contained dataset suitable for further analysis.

        Args:
            results (Dict[str, Any]): A dictionary containing the experiment's results.
                Must include the keys: 'X_test', 'y_test', 'prediction',
                'model_name', and 'metrics' (a dictionary of metric names to values).
            filename (str): The path and name for the output CSV file.

        Returns:
            pd.DataFrame: The comprehensive DataFrame that was written to the CSV file.

        Raises:
            ValueError: If the `results` dictionary is missing any required keys.
        """

        print(f"Exporting results to '{filename}'...")

        required_keys = ["X_test", "y_test", "prediction", "model_name", "metrics"]
        if not all(key in results for key in required_keys):
            raise ValueError(
                f"The 'results' dictionary must contain all keys: {required_keys}"
            )

        df_export = pd.DataFrame(
            {
                "X_test": results["X_test"],
                "y_test": results["y_test"],
                "prediction": results["prediction"],
            }
        )

        df_export["model_name"] = results["model_name"]

        # 3. Add each metric as its own column
        metrics_dict = results["metrics"]
        for metric_name, metric_value in metrics_dict.items():
            df_export[metric_name] = metric_value

        df_export.to_csv(filename, index=True)

        print(f"Successfully exported results to '{filename}'.")
        return df_export
