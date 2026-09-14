"""Sphinx configuration for streamgoggles."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "streamgoggles"
copyright = "2026, MatthieuPe"
author = "MatthieuPe"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary --------------------------------------------------
# Every module here docstrings itself with "Parameters:"/"Returns:"/
# "Raises:" sections (napoleon's Google-style parser recognizes
# "Parameters" as an alias for "Args"), plus a codebase-specific
# "Rationale:" paragraph that isn't a special napoleon section -- it just
# renders as ordinary body text, which is fine.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

napoleon_google_docstring = True
# A handful of older functions (data_preparation.py's process_data/
# process_data_polyfit2d) predate this codebase's Google-style convention
# and still use NumPy-style "Parameters\n----------" docstrings -- napoleon
# parses both simultaneously, so both stay correctly formatted without
# rewriting that legacy code.
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
# Every @dataclass here documents its fields in an "Attributes:" docstring
# section. With the default napoleon_use_ivar=False, that section becomes
# its own standalone ".. py:attribute::" entries, which collide with
# autodoc's own per-field member entries for the same @dataclass fields
# ("duplicate object description"). napoleon_use_ivar=True instead renders
# the section as a definition list inside the class description, so there's
# only ever one registered object per field.
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence", "deflist"]

# -- intersphinx --------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- lean CI builds: mock the heavy/data-dependent imports -------------
# Set only when STREAMGOGGLES_DOCS_LEAN is in the environment (the CI
# workflow sets it; a normal local build, with the real packages actually
# installed, does not) -- autodoc_mock_imports unconditionally substitutes
# a fake stand-in for these module names whenever it's set at all, even if
# the real package happens to be installed too, so this must stay opt-in
# rather than always-on or it would silently degrade a real local build.
#
# torch/streamobs/ugali/gala/astropy/scipy/matplotlib/hyrax are the
# packages actually driving CI install time (hyrax alone pulls in
# mlflow/chromadb/onnxruntime and more) and/or need real external
# survey/isochrone data to do anything beyond a bare import -- neither of
# which a docs build needs. numpy/pandas/healpy/pyyaml are NOT mocked even
# in lean mode: they're small, fast, dependency-light, and mocking numpy
# specifically breaks `x: np.random.Generator | None = ...`-style
# parameter annotations (Sphinx's mock objects don't support `|`, and this
# codebase evaluates those eagerly, having no `from __future__ import
# annotations` anywhere) -- confirmed by hitting exactly that TypeError
# while building this configuration out for real.
#
# Two API pages (models.losses, training.plain_runner) still end up empty
# in lean mode even with this split -- both use a bare `torch.Tensor |
# None` parameter annotation with no numpy equivalent trick available.
# Real, accepted gap in CI-built docs specifically; a local build (this
# env var unset) documents them fully.
if os.environ.get("STREAMGOGGLES_DOCS_LEAN"):
    autodoc_mock_imports = [
        "torch",
        "streamobs",
        "ugali",
        "gala",
        "astropy",
        "scipy",
        "matplotlib",
        "hyrax",
    ]

# -- HTML output --------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "streamgoggles"
