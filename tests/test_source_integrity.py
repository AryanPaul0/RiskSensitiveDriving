"""Check that easier explanations cannot hide a changed calculation."""

import hashlib
import json
from driving_rl.source_integrity import (
    executable_signature,
    source_matches_frozen_version,
)


def test_explanations_and_formatting_preserve_instructions(tmp_path):
    old = "def value():\n    return 2 + 3\n"
    new = '"""Explain this module."""\n# A teaching comment.\ndef value():\n    """Add two numbers."""\n    return 2 + 3\n'
    assert executable_signature(old) == executable_signature(new)
    path = tmp_path / "sample.py"
    path.write_text(new)
    results = tmp_path / "results"
    results.mkdir()
    locked = hashlib.sha256(old.encode()).hexdigest()
    audit = {
        "files": {
            "sample.py": {
                "before_sha256": locked,
                "executable_signature": executable_signature(old),
            }
        }
    }
    (results / "readability_audit.json").write_text(json.dumps(audit))
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)
    path.write_text(new.replace("2 + 3", "2 + 4"))
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)


def test_exact_source_needs_no_exception_and_unrecorded_change_fails(tmp_path):
    source = "answer = 5\n"
    (tmp_path / "sample.py").write_text(source)
    locked = hashlib.sha256(source.encode()).hexdigest()
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)
    (tmp_path / "sample.py").write_text("# New comment\n" + source)
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)


def test_random_draw_order_and_constants_are_preserved():
    original = "a = rng.random()\nb = rng.integers(5)\n"
    reordered = "b = rng.integers(5)\na = rng.random()\n"
    assert executable_signature(original) != executable_signature(reordered)
    assert executable_signature("gamma = .9") != executable_signature("gamma = .99")
