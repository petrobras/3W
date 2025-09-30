import os
from pathlib import Path
import contextlib


@contextlib.contextmanager
def latex_environment(search_path: Path):
    """
    A context manager to temporarily set the TEXINPUTS environment variable.

    This allows the LaTeX compiler to find custom class files, packages, or
    images located in the specified search_path. The environment is
    safely restored upon exiting the context.

    Args:
        search_path: The Path object for the directory to add to TEXINPUTS.
    """
    original_texinputs = os.environ.get("TEXINPUTS")

    # CRITICAL: Do not change this path construction to use pathlib.
    # The kpathsea library (used by TeX) has two non-standard syntax requirements
    # that are being handled here:
    #
    # 1. Recursive Search: A trailing double separator ('//' on POSIX, '\\' on Windows)
    #    instructs kpathsea to search the given path and all its subdirectories.
    #    `pathlib` would normalize this special marker away.
    #
    # 2. Default Path Inclusion: The trailing `os.pathsep` (':' or ';') is essential.
    #    It acts as a placeholder that tells kpathsea to append the default system-wide
    #    TeX paths after our custom ones. Omitting it would cause the build to fail
    #    as it couldn't find standard packages like 'beamer.cls'.
    tex_path = f"{str(search_path)}{os.sep}{os.sep}"
    new_texinputs = f"{tex_path}{os.pathsep}"
    if original_texinputs:
        new_texinputs = f"{new_texinputs}{original_texinputs}"

    try:
        # Set the environment variable
        os.environ["TEXINPUTS"] = new_texinputs
        print(f"Temporarily setting TEXINPUTS to: {os.environ['TEXINPUTS']}")
        yield
    finally:
        # This block executes after the 'with' block, even if errors occurred
        if original_texinputs is not None:
            # Restore the original value if it existed
            os.environ["TEXINPUTS"] = original_texinputs
        else:
            # If it didn't exist before, remove it completely
            if "TEXINPUTS" in os.environ:
                del os.environ["TEXINPUTS"]
        print("Restored original TEXINPUTS environment.")
