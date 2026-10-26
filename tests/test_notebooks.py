"""Static checks on the dev/ wrapper notebooks (they can't run without the raw data)."""

import ast
import importlib
import inspect

import nbformat
import pytest

from conftest import REPO

NOTEBOOKS = sorted((REPO / "dev").glob("*.ipynb"))


def _code(cell):
    return "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))


def test_one_wrapper_per_original():
    originals = sorted(p.name for p in (REPO / "dev" / "original_nbs").glob("*.ipynb"))
    assert [p.name for p in NOTEBOOKS] == originals


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)

    first = nb.cells[0]
    assert first.cell_type == "code" and "%load_ext autoreload" in first.source and "%autoreload 2" in first.source

    aliases = {}
    trees = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        assert not cell.get("outputs"), "outputs should be cleared"
        tree = ast.parse(_code(cell))
        trees.append(tree)
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)), \
                "logic belongs in the package"
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith(("wildfire_stats", "utils")) for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("wildfire_stats"):
                for a in node.names:
                    assert a.asname, f"use `from {node.module} import {a.name} as ...`"
                    try:
                        aliases[a.asname] = importlib.import_module(f"{node.module}.{a.name}")
                    except ModuleNotFoundError:
                        if ".old" in node.module or a.name == "old":
                            pytest.skip("wildfire_stats.old archive not present")
                        raise

    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                assert hasattr(aliases[node.value.id], node.attr), f"{node.value.id}.{node.attr} does not exist"
            # calls into the package must bind to the real signature
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases):
                fn = getattr(aliases[node.func.value.id], node.func.attr)
                if not callable(fn) or any(isinstance(a, ast.Starred) for a in node.args) \
                        or any(k.arg is None for k in node.keywords):
                    continue
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                sig.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})
