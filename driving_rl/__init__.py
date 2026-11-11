"""Prepare temporary folders for plotting libraries when this package is imported.

Python runs this small file before loading the other driving_rl modules. The
settings below keep graphics caches out of restricted personal folders.

Original implementation notes:
Small, inspectable deep reinforcement learning experiments.
"""

import os
from pathlib import Path
import tempfile

# Keep graphics caches out of the user's home directory in restricted runtimes.
_cache = Path(tempfile.gettempdir()) / "risk-aware-driving-cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
