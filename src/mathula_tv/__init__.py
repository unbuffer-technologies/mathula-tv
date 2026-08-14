"""Test-only namespace bridge for the partial source release workspace."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

__version__ = "0.1.3"
