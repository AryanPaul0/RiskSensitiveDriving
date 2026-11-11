"""Recognize explanation-only source edits without weakening model checks.

Saved experiments record file fingerprints. Adding comments changes those
fingerprints even when calculations stay the same. Here we compare Python's
parsed instructions instead: comments and teaching docstrings may differ, but
operators, constants, calls, their order, and random-number operations must agree.
This does not permit changing a frozen checkpoint or a scientific setting.
"""

import ast  # Make ast available for this file's calculations and workflow.
import hashlib  # Make hashlib available for this file's calculations and workflow.
import json  # Make json available for this file's calculations and workflow.
from pathlib import Path  # Make Path available for this file's calculations and workflow.


def executable_signature(source):  # Fingerprint parsed Python instructions, excluding teaching docstrings.
    """Fingerprint parsed Python instructions, excluding teaching docstrings.

    Python already ignores comments when it parses source. We also remove the
    first string in a module, class or function: that string is its docstring.
    Help descriptions can therefore improve while numerical instructions stay
    fixed. This check is for this project, whose docstrings are explanations.
    """
    tree = ast.parse(source)  # Set tree to ast.parse applied to source.
    for node in ast.walk(tree):  # Process each node from ast.walk applied to tree.
        if isinstance(  # Use this branch when isinstance applied to node, ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef.
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)  # Include node, ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef in the surrounding operation.
        ):  # Finish assembling isinstance inputs.
            if node.body:  # Use this branch when node.body.
                first = node.body[0]  # Set first to the entries of node.body selected by 0.
                if (  # Use this branch when isinstance applied to first, ast.Expr and isinstance applied to first.value, ast.Constant and isinstance applied to first.value.value, str.
                    isinstance(first, ast.Expr)  # Include isinstance applied to first, ast.Expr and isinstance applied to first.value, ast.Constant and isinstance applied to first.value.value, str in the surrounding operation.
                    and isinstance(first.value, ast.Constant)  # Include isinstance applied to first.value, ast.Constant in the surrounding operation.
                    and isinstance(first.value.value, str)  # Include isinstance applied to first.value.value, str in the surrounding operation.
                ):  # Finish assembling the grouped values above.
                    node.body = node.body[1:]  # Set node.body to the entries of node.body selected by 1:.
    instructions = ast.dump(tree, include_attributes=False)  # Set instructions to ast.dump applied to tree.
    return hashlib.sha256(instructions.encode()).hexdigest()  # Return the result of hashlib.sha256(instructions.encode()).hexdigest.


def source_matches_frozen_version(root, relative_name, locked_hash):  # Accept the exact old file, or a recorded explanation-only equivalent.
    """Accept the exact old file, or a recorded explanation-only equivalent.

    The audit links an original byte fingerprint to its executable signature.
    A real calculation change must still fail, even if comments were also added.
    Model checkpoint hashes are checked separately and are never exempted here.
    """
    root = Path(root)  # Set root to Path applied to root.
    source = (root / relative_name).read_text()  # Set source to the result of (root / relative name).read text.
    if hashlib.sha256(source.encode()).hexdigest() == locked_hash:  # Use this branch when the result of hashlib.sha256(source.encode()).hexdigest equals locked hash.
        return True  # Return True.
    audit_path = root / "results/readability_audit.json"  # Set audit path to the path to 'results/readability_audit.json' inside root.
    if not audit_path.exists():  # Use this branch when not the result of audit path.exists.
        return False  # Return False.
    record = json.loads(audit_path.read_text()).get("files", {}).get(relative_name)  # Set record to json.loads(audit path.read text()).get('files', {}).get applied to relative name.
    if not record or record["before_sha256"] != locked_hash:  # Use this branch when not record or the entries of record selected by 'before_sha256' differs from locked hash.
        return False  # Return False.
    return executable_signature(source) == record["executable_signature"]  # Return executable signature applied to source equals the entries of record selected by 'executable_signature'.
