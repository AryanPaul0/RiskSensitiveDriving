# This quoted explanation is a docstring when it starts a function, class, or module; help() can display it.
"""Check that easier explanations cannot hide a changed calculation."""

# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from driving_rl.source_integrity import brings the named tools into this file; parentheses allow several lines.
from driving_rl.source_integrity import (
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    executable_signature,
    # This is one item in the expression opened above; commas separate items, and indentation helps us read their grouping.
    source_matches_frozen_version,
# ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
)


# def defines test_explanations_and_formatting_preserve_instructions; parentheses list its inputs, and the colon starts its indented body.
def test_explanations_and_formatting_preserve_instructions(tmp_path):
    # = stores a result in old; use quoted text as data; Python does not execute the words inside it.
    old = "def value():\n    return 2 + 3\n"
    # = stores a result in new; use quoted text as data; Python does not execute the words inside it.
    new = '"""Explain this module."""\n# A teaching comment.\ndef value():\n    """Add two numbers."""\n    return 2 + 3\n'
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert executable_signature(old) == executable_signature(new)
    # = stores a result in path; apply / to these operands; numbers are divided, while Path objects join folder names.
    path = tmp_path / "sample.py"
    # Call path.write_text using parentheses to supply its inputs; this line does not store a separate return value.
    path.write_text(new)
    # = stores a result in results; apply / to these operands; numbers are divided, while Path objects join folder names.
    results = tmp_path / "results"
    # Call results.mkdir using parentheses to supply its inputs; this line does not store a separate return value.
    results.mkdir()
    # = stores a result in locked; call hashlib.sha256(old.encode()).hexdigest using parentheses to supply its inputs.
    locked = hashlib.sha256(old.encode()).hexdigest()
    # = stores a result in audit; build a dictionary, which stores values under named keys.
    audit = {
        # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
        "files": {
            # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
            "sample.py": {
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "before_sha256": locked,
                # The colon pairs this dictionary key with its value; a trailing comma separates it from the next entry.
                "executable_signature": executable_signature(old),
            # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
            }
        # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
        }
    # } closes the dictionary, set, or formatted-string expression opened above; any following comma separates this item from the next.
    }
    # Call (results / 'readability_audit.json').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (results / "readability_audit.json").write_text(json.dumps(audit))
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)
    # Call path.write_text using parentheses to supply its inputs; this line does not store a separate return value.
    path.write_text(new.replace("2 + 3", "2 + 4"))
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)


# def defines test_exact_source_needs_no_exception_and_unrecorded_change_fails; parentheses list its inputs, and the colon starts its indented body.
def test_exact_source_needs_no_exception_and_unrecorded_change_fails(tmp_path):
    # = stores a result in source; use quoted text as data; Python does not execute the words inside it.
    source = "answer = 5\n"
    # Call (tmp_path / 'sample.py').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (tmp_path / "sample.py").write_text(source)
    # = stores a result in locked; call hashlib.sha256(source.encode()).hexdigest using parentheses to supply its inputs.
    locked = hashlib.sha256(source.encode()).hexdigest()
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)
    # Call (tmp_path / 'sample.py').write_text using parentheses to supply its inputs; this line does not store a separate return value.
    (tmp_path / "sample.py").write_text("# New comment\n" + source)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)


# def defines test_random_draw_order_and_constants_are_preserved; parentheses list its inputs, and the colon starts its indented body.
def test_random_draw_order_and_constants_are_preserved():
    # = stores a result in original; use quoted text as data; Python does not execute the words inside it.
    original = "a = rng.random()\nb = rng.integers(5)\n"
    # = stores a result in reordered; use quoted text as data; Python does not execute the words inside it.
    reordered = "b = rng.integers(5)\na = rng.random()\n"
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert executable_signature(original) != executable_signature(reordered)
    # assert checks this condition; a false result raises AssertionError and fails this check.
    assert executable_signature("gamma = .9") != executable_signature("gamma = .99")
