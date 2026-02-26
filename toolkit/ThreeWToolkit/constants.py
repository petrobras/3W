from pathlib import Path
from importlib import resources
from datetime import datetime
from .logging_config import setup_default_logging

package = "ThreeWToolkit"
# We define the project root as the directory where the toolkit is installed (i.e. the parent directory of ThreeWToolkit).
# This makes all other paths relative to the project's top-level folder.
PROJECT_ROOT = Path(str(resources.files(package))).parent

# Source final output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"  # Directory for generated PDFs

# Define the path to the mock visualization objects plot folder
PLOTS_DIR = OUTPUT_DIR / "3w_plots"

LOGS_DIR = OUTPUT_DIR / "logs"
_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
setup_default_logging(LOGS_DIR, run_id=_run_id)

# Define the path to the LaTeX and HTML template documents directory
REPORTS_DIR = PROJECT_ROOT / package / "reports"
LATEX_DIR = REPORTS_DIR / "latex"
HTML_TEMPLATES_DIR = REPORTS_DIR / "html"
CSS_PATH = HTML_TEMPLATES_DIR / "petro.css"
HTML_ASSETS_DIR = HTML_TEMPLATES_DIR / "assets"

# Define the path to the dataset 2.0.0 config
DATASET_INI_2_0_0 = (
    Path(str(resources.files(package))).parents[1] / "dataset/dataset.ini"
)
