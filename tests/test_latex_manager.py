from ThreeWToolkit.utils.latex_manager import copy_latex_support_files
from pathlib import Path


def test_copy_all_files_successfully(tmp_path: Path):
    """
    Tests that .sty files and the assets directory are copied correctly
    to an empty destination.
    """
    # 1. Setup: Create source and destination directories
    latex_dir = tmp_path / "latex_src"
    latex_dir.mkdir()
    report_path = tmp_path / "report_dest"
    report_path.mkdir()

    # Create dummy source files
    (latex_dir / "style1.sty").touch()
    (latex_dir / "style2.sty").touch()
    (latex_dir / "other_file.txt").touch()  # This should be ignored

    # Create source assets directory
    assets_src = latex_dir / "assets"
    assets_src.mkdir()
    (assets_src / "logo.png").touch()

    # 2. Act: Call the function
    copy_latex_support_files(latex_dir, report_path)

    # 3. Assert: Check if files and directories were copied correctly
    assert (report_path / "style1.sty").exists()
    assert (report_path / "style2.sty").exists()
    assert not (
        report_path / "other_file.txt"
    ).exists()  # Ensure non-.sty files aren't copied

    assets_dest = report_path / "assets"
    assert assets_dest.is_dir()
    assert (assets_dest / "logo.png").exists()


def test_copy_with_existing_assets_directory(tmp_path: Path):
    """
    Tests that an existing 'assets' directory in the destination is
    correctly replaced.
    """
    # 1. Setup
    latex_dir = tmp_path / "latex_src"
    latex_dir.mkdir()
    report_path = tmp_path / "report_dest"
    report_path.mkdir()

    # Create source assets with a new file
    assets_src = latex_dir / "assets"
    assets_src.mkdir()
    (assets_src / "new_logo.svg").touch()

    # Create a pre-existing assets directory in the destination with an old file
    assets_dest = report_path / "assets"
    assets_dest.mkdir()
    (assets_dest / "old_logo.png").touch()

    # 2. Act
    copy_latex_support_files(latex_dir, report_path)

    # 3. Assert
    # The new assets directory should exist
    assert (report_path / "assets").is_dir()
    # The new file should be there
    assert (report_path / "assets" / "new_logo.svg").exists()
    # The old file should be gone
    assert not (report_path / "assets" / "old_logo.png").exists()


def test_no_sty_files_to_copy(tmp_path: Path):
    """
    Tests that the function runs without error when no .sty files are present.
    """
    # 1. Setup
    latex_dir = tmp_path / "latex_src"
    latex_dir.mkdir()
    report_path = tmp_path / "report_dest"
    report_path.mkdir()

    # Create only an assets directory
    assets_src = latex_dir / "assets"
    assets_src.mkdir()
    (assets_src / "image.jpg").touch()

    # 2. Act
    copy_latex_support_files(latex_dir, report_path)

    # 3. Assert
    assert (report_path / "assets" / "image.jpg").exists()
    # Check that no stray .sty files were created
    assert not list(report_path.glob("*.sty"))


def test_no_assets_directory_to_copy(tmp_path: Path):
    """
    Tests that the function runs without error when no 'assets' directory is present.
    """
    # 1. Setup
    latex_dir = tmp_path / "latex_src"
    latex_dir.mkdir()
    report_path = tmp_path / "report_dest"
    report_path.mkdir()

    (latex_dir / "awesome.sty").touch()

    # 2. Act
    copy_latex_support_files(latex_dir, report_path)

    # 3. Assert
    assert (report_path / "awesome.sty").exists()
    assert not (report_path / "assets").exists()


# -----------------------------------------------------------------------------
# Test for exception handling
# -----------------------------------------------------------------------------
def test_source_directory_does_not_exist(tmp_path: Path, capsys):
    """
    Tests that the function handles a non-existent source directory gracefully
    by catching the exception and printing a warning.
    """
    # 1. Setup
    non_existent_dir = tmp_path / "non_existent"
    report_path = tmp_path / "report_dest"
    report_path.mkdir()

    # 2. Act
    copy_latex_support_files(non_existent_dir, report_path)

    # 3. Assert
    captured = capsys.readouterr()
    # Check that the warning message was printed to stdout
    assert "Warning: Could not copy LaTeX support files" in captured.out
    # Check that the destination directory remains empty
    assert not list(report_path.iterdir())
