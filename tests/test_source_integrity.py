"""Check that easier explanations cannot hide a changed calculation."""  # Describe this operation for readers and help().

import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from driving_rl.source_integrity import (  # Make executable_signature, source_matches_frozen_version available for this file's calculations and workflow.
    executable_signature,  # Continue the calculation or input list started above.
    source_matches_frozen_version,  # Continue the calculation or input list started above.
)  # Finish assembling the grouped values above.


def test_explanations_and_formatting_preserve_instructions(tmp_path):  # Define the test explanations and formatting preserve instructions operation used by this workflow.
    old = "def value():\n    return 2 + 3\n"  # Set old to 'def value():\n return 2 + 3\n'.
    new = '"""Explain this module."""\n# A teaching comment.\ndef value():\n    """Add two numbers."""\n    return 2 + 3\n'  # Set new to the report text.
    assert executable_signature(old) == executable_signature(new)  # Check that executable signature applied to old equals executable signature applied to new; fail this test if it is not true.
    path = tmp_path / "sample.py"  # Set path to the path to 'sample.py' inside tmp path.
    path.write_text(new)  # Path.write text applied to new.
    results = tmp_path / "results"  # Set results to the path to 'results' inside tmp path.
    results.mkdir()  # The result of results.mkdir.
    locked = hashlib.sha256(old.encode()).hexdigest()  # Set locked to the result of hashlib.sha256(old.encode()).hexdigest.
    audit = {  # Set audit to the named settings or measurements assembled below.
        "files": {  # Record 'files' as the named settings or measurements assembled below.
            "sample.py": {  # Record 'sample.py' as the named settings or measurements assembled below.
                "before_sha256": locked,  # Record 'before_sha256' as locked.
                "executable_signature": executable_signature(old),  # Record 'executable_signature' as executable signature applied to old.
            }  # Finish assembling the grouped values above.
        }  # Finish assembling the grouped values above.
    }  # Finish assembling the grouped values above.
    (results / "readability_audit.json").write_text(json.dumps(audit))  # (results / 'readability audit.json').write text applied to json.dumps applied to audit.
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)  # Check that source matches frozen version applied to tmp path, 'sample.py', locked; fail this test if it is not true.
    path.write_text(new.replace("2 + 3", "2 + 4"))  # Path.write text applied to new.replace applied to '2 + 3', '2 + 4'.
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)  # Check that not source matches frozen version applied to tmp path, 'sample.py', locked; fail this test if it is not true.


def test_exact_source_needs_no_exception_and_unrecorded_change_fails(tmp_path):  # Define the test exact source needs no exception and unrecorded change fails operation used by this workflow.
    source = "answer = 5\n"  # Set source to 'answer = 5\n'.
    (tmp_path / "sample.py").write_text(source)  # (tmp path / 'sample.py').write text applied to source.
    locked = hashlib.sha256(source.encode()).hexdigest()  # Set locked to the result of hashlib.sha256(source.encode()).hexdigest.
    assert source_matches_frozen_version(tmp_path, "sample.py", locked)  # Check that source matches frozen version applied to tmp path, 'sample.py', locked; fail this test if it is not true.
    (tmp_path / "sample.py").write_text("# New comment\n" + source)  # (tmp path / 'sample.py').write text applied to ('# new comment\n' plus source).
    assert not source_matches_frozen_version(tmp_path, "sample.py", locked)  # Check that not source matches frozen version applied to tmp path, 'sample.py', locked; fail this test if it is not true.


def test_random_draw_order_and_constants_are_preserved():  # Define the test random draw order and constants are preserved operation used by this workflow.
    original = "a = rng.random()\nb = rng.integers(5)\n"  # Set original to 'a = rng.random()\nb = rng.integers(5)\n'.
    reordered = "b = rng.integers(5)\na = rng.random()\n"  # Set reordered to 'b = rng.integers(5)\na = rng.random()\n'.
    assert executable_signature(original) != executable_signature(reordered)  # Check that executable signature applied to original differs from executable signature applied to reordered; fail this test if it is not true.
    assert executable_signature("gamma = .9") != executable_signature("gamma = .99")  # Check that executable signature applied to 'gamma = .9' differs from executable signature applied to 'gamma = .99'; fail this test if it is not true.
