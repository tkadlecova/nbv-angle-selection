"""Put the repository's script folders on sys.path, so that a script in any
folder can import modules from the others (core, experiments, observer)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("core", "experiments", "observer"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.append(_p)
