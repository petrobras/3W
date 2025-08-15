from pathlib import Path

# We define the project root as the directory where the toolkit is installed (i.e. the parent directory of ThreeWToolkit).
# This makes all other paths relative to the project's top-level folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source final output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"  # Directory for generated PDFs

# Define the path to the mock visualization objects plot folder
PLOTS_DIR = Path("/tmp/3w_plots")

# Define the path to the LaTeX documents directory
LATEX_DIR = PROJECT_ROOT / "docs" / "latex"
