"""Prepare temporary folders for plotting libraries when this package is imported.

Python runs this small file before loading the other driving_rl modules. The
settings below keep graphics caches out of restricted personal folders.

Original implementation notes:
Small, inspectable deep reinforcement learning experiments.
"""

import os  # Make os available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.
import tempfile  # Make tempfile available for this file's calculations and workflow.

_cache = Path(tempfile.gettempdir()) / "risk-aware-driving-cache"  # Keep graphics caches out of the user's home directory in restricted runtimes.
_cache.mkdir(parents=True, exist_ok=True)  # The result of cache.mkdir.
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))  # Os.environ.setdefault applied to 'mplconfigdir', the path to 'matplotlib' inside cache as a text value.
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))  # Os.environ.setdefault applied to 'xdg_cache_home', cache as a text value.
