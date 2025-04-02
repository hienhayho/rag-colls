# Configuration file for the Sphinx documentation builder.

# -- Project information
import os

project = "rag-colls"
copyright = "2025, hienhayho"
author = "hienhayho"

release = "0.2.0.1"
version = "0.2.0.1"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
# html_theme = "furo"
html_theme = "sphinx_book_theme"

html_logo = "_static/Final_logo.png"

html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"
html_js_files = [
    ("readthedocs.js", {"defer": "defer"}),
]
