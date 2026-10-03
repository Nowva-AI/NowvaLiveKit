"""Shared fixtures for the agent-side tests."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import pytest

MAIN_PY = Path(__file__).parent.parent / "src" / "main.py"


def _compile_main_function(name: str) -> Callable:
    tree = ast.parse(MAIN_PY.read_text())
    namespace: dict = {}
    for node in tree.body:
        is_constant = isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper() and isinstance(node.value, ast.Constant)
            for target in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name == name):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
    return namespace[name]


@pytest.fixture
def main_function() -> Callable[[str], Callable]:
    """Compiles one top-level function of main.py (with its module constants); importing
    main.py itself starts the database and the voice stack."""
    return _compile_main_function
