# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import subprocess


def run_generate_examples():
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(docs_dir, "generate_examples.py")
    subprocess.run(["python", script_path], check=True)


run_generate_examples()

project = "rag-colls"
copyright = "2025, hienhayho"
author = "rag-colls team."

release = "0.2.0.6"
version = "0.2.0.6"

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

html_logo = "_static/rag_colls_v3.png"
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/hienhayho/rag-colls",
    "use_repository_button": True,
    "use_edit_page_button": True,
}

html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"
html_js_files = [
    ("readthedocs.js", {"defer": "defer"}),
]
