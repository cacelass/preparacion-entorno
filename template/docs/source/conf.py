import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = '{{ project_slug }}'
copyright = '2026, {{ project_author_name }}'
author = '{{ project_author_name }}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
