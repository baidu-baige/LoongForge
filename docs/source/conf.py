import os, sys
sys.path.insert(0, os.path.abspath('../..')) 


project = 'LoongForge'
copyright = '2026, LoongForge'
author = 'LoongForge'

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "substitution",
    "tasklist",
    "colon_fence",
]

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]
html_show_sphinx = False
html_show_copyright = True

