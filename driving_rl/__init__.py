"""Prepare temporary folders for plotting libraries when this package is imported.

Python runs this small file before loading the other driving_rl modules. The
settings below keep graphics caches out of restricted personal folders.

Original implementation notes:
Small, inspectable deep reinforcement learning experiments.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import os
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path
# import makes these modules available; "as" gives a module a shorter local name.
import tempfile

# Keep graphics caches out of the user's home directory in restricted runtimes.
# = stores a result in _cache; apply / to these operands; numbers are divided, while Path objects join folder names.
_cache = Path(tempfile.gettempdir()) / "risk-aware-driving-cache"
# Call _cache.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
_cache.mkdir(parents=True, exist_ok=True)
# Call os.environ.setdefault using parentheses to supply its inputs; this line does not store a separate return value.
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
# Call os.environ.setdefault using parentheses to supply its inputs; this line does not store a separate return value.
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))
