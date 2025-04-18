# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Transferability Survey'
copyright = '2025, Haohua Wang, Jingge Wang, Zijie Zhao, Yang Li'
author = 'Haohua Wang, Jingge Wang, Zijie Zhao, Yang Li'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'recommonmark', 
    'sphinx_markdown_tables', 
    'sphinxcontrib.bibtex', 
    'myst_parser']

templates_path = ['_templates']
exclude_patterns = []
bibtex_bibfiles = ['md/references.bib']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
html_static_path = ['sphinx_rtd_theme.get_html_theme_path()']


