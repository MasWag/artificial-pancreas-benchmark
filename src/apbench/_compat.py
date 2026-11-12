"""Import-time compatibility shims for Simglucose's old dependencies.

Simglucose and its dependency ``gym==0.9.4`` import ``pkg_resources``, which
Setuptools removed in version 81; hence the ``setuptools<81`` requirement.
This module only silences the resulting deprecation and syntax warnings.
"""

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"gym(\.|$)")

try:
    import pkg_resources  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Simglucose requires pkg_resources; install setuptools<81 (see README, 'Installation')"
    ) from e
