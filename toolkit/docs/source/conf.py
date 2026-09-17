# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import shutil

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ThreeWToolkit"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ThreeWToolkit"
copyright = "2026, Ricardo Emanuel Vaz Vargas"
author = "Ricardo Emanuel Vaz Vargas"
release = "3.2.1"
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

autosummary_generate = True
add_module_names = False
python_use_unqualified_type_names = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

nbsphinx_execute = "never"

nitpick_ignore = [
    ("py:class", "DatasetOutputs"),
    ("py:class", "Path"),
]

nitpick_ignore_regex = [
    (r"py:.*", r"pydantic\.functional_validators\.field_validator"),
]

suppress_warnings = ["ref.python", "ref.footnote", "ref.ref", "ref.doc"]


def copy_demos(app):
    source_dir = Path(app.srcdir)

    demos_src = source_dir.parent.parent / "demos"
    demos_dst = source_dir / "_demos_gen"

    print(f"[ThreeWToolkit] Demos source: {demos_src}")
    print(f"[ThreeWToolkit] Demos destination: {demos_dst}")

    if not demos_src.exists():
        print(f"[ThreeWToolkit] Demos directory not found: {demos_src}")
        return

    if demos_dst.exists():
        shutil.rmtree(demos_dst)

    shutil.copytree(demos_src, demos_dst, dirs_exist_ok=True)

    print("[ThreeWToolkit] Demos copied successfully")


def setup(app):
    app.connect("config-inited", lambda app, config: copy_demos(app))
