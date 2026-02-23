try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("ThreeWToolkit")
except PackageNotFoundError:
    # Package is not installed, try to read from pyproject.toml
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
            __version__ = pyproject["project"]["version"]
    else:
        __version__ = "unknown"
