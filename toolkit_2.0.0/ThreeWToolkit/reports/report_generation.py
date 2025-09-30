import pandas as pd

from pathlib import Path
from typing import Any, List, Dict

from pylatex import Document, Section, Command, Center, Itemize
from pylatex.utils import NoEscape
from pylatex.package import Package

from ThreeWToolkit.data_visualization.plot_series import DataVisualization

from ..constants import LATEX_DIR, REPORTS_DIR
from ..utils.latex_manager import latex_environment
from ..metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    explained_variance_score,
)


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
        metrics: List[str],
        predictions: pd.Series,
        calculated_metrics: dict,
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
        self.metrics = metrics
        self.predictions = predictions
        self.calculated_metrics = calculated_metrics
        self.title = title
        self.author = author
        self.latex_dir = latex_dir
        self.reports_dir = reports_dir
        self.save_report_after_generate = export_report_after_generate

        # Default mapping (pode ser expandido ou sobrescrito depois)
        self.metric_function_map = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "recall": recall_score,
            "precision": precision_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
            "average_precision": average_precision_score,
            "get_explained_variance": explained_variance_score,
        }

    def _format_metric_name(self, method_name: str) -> str:
        if method_name == "f1":
            return "F1 Score"
        if method_name == "get_roc_auc":
            return "ROC AUC"
        return method_name.replace("get_", "").replace("_", " ").strip().title()

    def generate_summary_report(self) -> Document:
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
                    doc.append(NoEscape(f"{name} & {value} \\\\"))
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
            # Plot 1: Predicted vs Actual
            doc.append(NoEscape(r"\begin{frame}{Visualization: Predicted vs. Actual}"))
            img_path = DataVisualization.plot_multiple_series(
                [self.y_test, self.predictions],
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
            fft_img_path = DataVisualization.plot_fft(self.y_test)
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
                self.y_test, period=20
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
                    "Actual": self.y_test,
                    "Predicted": self.predictions,
                    "Residuals": self.y_test - self.predictions,
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
            wavelet_img_path = DataVisualization.plot_wavelet_spectrogram(self.y_test)
            doc.append(NoEscape(r"\begin{figure}\centering"))
            doc.append(
                NoEscape(
                    f"\\includegraphics[width=0.8\\textwidth]{{{wavelet_img_path}}}"
                )
            )
            doc.append(NoEscape(r"\caption{Wavelet Spectrogram of the Signal}"))
            doc.append(NoEscape(r"\end{figure}"))
            doc.append(NoEscape(r"\end{frame}"))

        print("Beamer document generated successfully.")
        if self.save_report_after_generate:
            self.save_report(doc=doc, filename=self.title)

        return doc

    def save_report(self, doc: Document, filename: str) -> None:
        """Compiles and saves a PyLaTeX Document to a PDF file using lualatex."""
        with latex_environment(self.latex_dir):
            report_folder = f"report-{filename}"
            report_path = self.reports_dir / report_folder
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
        """Exports results to CSV file."""
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

        for metric_name, metric_value in results["metrics"].items():
            df_export[metric_name] = metric_value

        df_export.to_csv(filename, index=True)
        print(f"Successfully exported results to '{filename}'.")
        return df_export
