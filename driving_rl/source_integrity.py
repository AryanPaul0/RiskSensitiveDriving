"""Recognize explanation-only source edits without weakening model checks.

Saved experiments record file fingerprints. Adding comments changes those
fingerprints even when calculations stay the same. Here we compare Python's
parsed instructions instead: comments and teaching docstrings may differ, but
operators, constants, calls, their order, and random-number operations must agree.
This does not permit changing a frozen checkpoint or a scientific setting.
"""

# import makes these modules available; "as" gives a module a shorter local name.
import ast
# import makes these modules available; "as" gives a module a shorter local name.
import hashlib
# import makes these modules available; "as" gives a module a shorter local name.
import json
# from pathlib import brings the named tools into this file; parentheses allow several lines.
from pathlib import Path


# def defines executable_signature; parentheses list its inputs, and the colon starts its indented body.
def executable_signature(source):
    """Fingerprint parsed Python instructions, excluding teaching docstrings.

    Python already ignores comments when it parses source. We also remove the
    first string in a module, class or function: that string is its docstring.
    Help descriptions can therefore improve while numerical instructions stay
    fixed. This check is for this project, whose docstrings are explanations.
    """
    # = stores a result in tree; call ast.parse using parentheses to supply its inputs.
    tree = ast.parse(source)
    # for takes each item from ast.walk(tree) in turn; node holds the current item.
    for node in ast.walk(tree):
        # if checks isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)); only a true condition runs the indented block.
        if isinstance(
            # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
        ):
            # if checks node.body; only a true condition runs the indented block.
            if node.body:
                # = stores a result in first; use square brackets to select part of node.body.
                first = node.body[0]
                # if checks isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value....; only a true condition runs the indented block.
                if (
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    isinstance(first, ast.Expr)
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    and isinstance(first.value, ast.Constant)
                    # Parentheses call the named helper or group an expression; this line supplies part of the surrounding expression.
                    and isinstance(first.value.value, str)
                # ) closes the parenthesized expression or function call opened above; any following comma separates this item from the next.
                ):
                    # = stores a result in node.body; use square brackets to select part of node.body.
                    node.body = node.body[1:]
    # = stores a result in instructions; call ast.dump using parentheses to supply its inputs.
    instructions = ast.dump(tree, include_attributes=False)
    # return ends this function and sends its result back to the code that called it.
    return hashlib.sha256(instructions.encode()).hexdigest()


# def defines source_matches_frozen_version; parentheses list its inputs, and the colon starts its indented body.
def source_matches_frozen_version(root, relative_name, locked_hash):
    """Accept the exact old file, or a recorded explanation-only equivalent.

    The audit links an original byte fingerprint to its executable signature.
    A real calculation change must still fail, even if comments were also added.
    Model checkpoint hashes are checked separately and are never exempted here.
    """
    # = stores a result in root; call Path using parentheses to supply its inputs.
    root = Path(root)
    # = stores a result in source; call (root / relative_name).read_text using parentheses to supply its inputs.
    source = (root / relative_name).read_text()
    # if checks hashlib.sha256(source.encode()).hexdigest() == locked_hash; only a true condition runs the indented block.
    if hashlib.sha256(source.encode()).hexdigest() == locked_hash:
        # return ends this function and sends its result back to the code that called it.
        return True
    # = stores a result in audit_path; apply / to these operands; numbers are divided, while Path objects join folder names.
    audit_path = root / "results/readability_audit.json"
    # if checks not audit_path.exists(); only a true condition runs the indented block.
    if not audit_path.exists():
        # return ends this function and sends its result back to the code that called it.
        return False
    # = stores a result in record; call json.loads(audit_path.read_text()).get('files', {}).get using parentheses to supply its inputs.
    record = json.loads(audit_path.read_text()).get("files", {}).get(relative_name)
    # if checks not record or record['before_sha256'] != locked_hash; only a true condition runs the indented block.
    if not record or record["before_sha256"] != locked_hash:
        # return ends this function and sends its result back to the code that called it.
        return False
    # return ends this function and sends its result back to the code that called it.
    return executable_signature(source) == record["executable_signature"]
