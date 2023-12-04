#!/usr/bin/env python
# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys

sys.path.insert(0, os.path.abspath('../../../../'))
print("PATH Direc:",os.getcwd())
print("LS:",os.listdir('./'))
# -- Project information -----------------------------------------------------

project = 'sconce'
copyright = '2023, Sathyaprakash'
author = 'Sathyaprakash'

release = '0.57'
version = '0.57.0'

# -- General configuration
html_logo = "https://github.com/satabios/sconce/blob/master/docs/source/images/sconce-punch-bk_removed.png?raw=true"

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]
html_logo = "images/sconce-punch-bk_removed.png"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_css_files = ['my_theme.css']

# -- Options for EPUB output
epub_show_urls = 'footnote'
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../../sconce/'))
# project = 'sconce'
# copyright = '2023, Sathyaprakash'
# author = 'Sathyaprakash'
#
# release = '0.73'
# version = '0.73.0'
#
# # -- General configuration
# html_logo = "https://github.com/satabios/sconce/blob/master/docs/source/images/sconce-punch-bk_removed.png?raw=true"
#
# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.intersphinx',
# ]
# html_logo = "images/sconce-punch-bk_removed.png"
#
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
# }
# intersphinx_disabled_domains = ['std']
#
# templates_path = ['_templates']
#
# # -- Options for HTML output
#
# html_theme = 'sphinx_rtd_theme'
#
# # -- Options for EPUB output
# epub_show_urls = 'footnote'
#
#
# html_css_files = ['my_theme.css']
