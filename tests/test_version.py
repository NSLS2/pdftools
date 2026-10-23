from importlib.metadata import version

import pdftools
from pdftools._version import __version__


def test_package_imports():
    assert pdftools.__name__ == "pdftools"


def test_version_matches_installed_metadata():
    assert __version__ == version("pdftools")
