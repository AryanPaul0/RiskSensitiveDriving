"""Recognize explanation-only source edits without weakening model checks.

Saved experiments record file fingerprints. Adding comments changes those
fingerprints even when calculations stay the same. Here we compare Python's
parsed instructions instead: comments and teaching docstrings may differ, but
operators, constants, calls, their order, and random-number operations must agree.
This does not permit changing a frozen checkpoint or a scientific setting.
"""

import ast
import hashlib
import json
from pathlib import Path


def executable_signature(source):
    """Fingerprint parsed Python instructions, excluding teaching docstrings.

    Python already ignores comments when it parses source. We also remove the
    first string in a module, class or function: that string is its docstring.
    Help descriptions can therefore improve while numerical instructions stay
    fixed. This check is for this project, whose docstrings are explanations.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    node.body = node.body[1:]
    instructions = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(instructions.encode()).hexdigest()


def source_matches_frozen_version(root, relative_name, locked_hash):
    """Accept the exact old file, or a recorded explanation-only equivalent.

    The audit links an original byte fingerprint to its executable signature.
    A real calculation change must still fail, even if comments were also added.
    Model checkpoint hashes are checked separately and are never exempted here.
    """
    root = Path(root)
    source = (root / relative_name).read_text()
    if hashlib.sha256(source.encode()).hexdigest() == locked_hash:
        return True
    audit_path = root / "results/readability_audit.json"
    if not audit_path.exists():
        return False
    record = json.loads(audit_path.read_text()).get("files", {}).get(relative_name)
    if not record or record["before_sha256"] != locked_hash:
        return False
    return executable_signature(source) == record["executable_signature"]
