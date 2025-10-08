from pathlib import Path
import shutil


def copy_latex_support_files(latex_dir: Path, report_path: Path):
    """
    Copies necessary .sty files and asset files for compilation from a source
    directory to a report directory.

    Args:
        latex_dir: The source directory containing .sty files and an 'assets' subdirectory.
        report_path: The destination directory for the report.
    """
    try:
        # Copy necessary .sty and asset files for compilation
        sty_files = list(latex_dir.glob("*.sty"))
        assets_dir_src = latex_dir / "assets"

        if not sty_files and not assets_dir_src.is_dir():
            raise FileNotFoundError(
                "No .sty files or 'assets' directory found to copy."
            )

        for sty_file in sty_files:
            shutil.copy(sty_file, report_path)

        if assets_dir_src.is_dir():
            assets_dir_dest = report_path / "assets"
            if assets_dir_dest.exists():
                shutil.rmtree(assets_dir_dest)
            shutil.copytree(assets_dir_src, assets_dir_dest)

        # print("Copied .sty files and assets to the report directory to facilitate user compilation.")

    except Exception as e:
        print(f"Warning: Could not copy LaTeX support files: {e}")
