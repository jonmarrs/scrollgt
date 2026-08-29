"""The packaged version must match what the package declares, and both must be real.

The package sat at 0.1.0 while the README announced v0.2, v0.3 and v0.3.1, so
`pip show scrollgt` reported 0.1.0 for a tool the documentation called v0.3.1.
For a benchmark whose whole purpose is that other people install and run it, a
version that does not identify the code is a real defect, not cosmetics: a bug
report saying "0.1.0" would be unactionable.

Pinned rather than noted, because a version drifts silently by construction.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import scrollgt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version():
    with open(os.path.join(ROOT, "pyproject.toml")) as fh:
        for line in fh:
            m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
            if m:
                return m.group(1)
    raise AssertionError("no version in pyproject.toml")


def test_package_and_pyproject_versions_agree():
    assert scrollgt.__version__ == _pyproject_version()


def test_version_is_not_the_stale_default():
    """0.1.0 is the value that persisted through three documented releases."""
    assert scrollgt.__version__ != "0.1.0"
