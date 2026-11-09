"""The research core stays free of the agent framework and of data vendors.

Two checks, because each misses what the other sees: the AST scan names the
offending line, and the fresh interpreter catches a transitive import that no
single file shows. The second runs in a subprocess because this process has
imported tradingagents and yfinance already (tests/conftest.py).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "tradingagents_quant"

# The one upstream module allowed to reach into the research core: the
# deterministic risk gate is wired into the agent graph there (milestone M7).
ALLOWED_UPSTREAM_SEAMS = {"tradingagents/graph/setup.py"}

# Optional web stack of the dashboard extra; its modules are skipped when absent.
_OPTIONAL = {"fastapi", "uvicorn", "starlette"}


def _forbidden(module: str) -> bool:
    root = module.split(".")[0]
    return root in {"tradingagents", "yfinance", "dotenv", "curl_cffi"} or root.startswith(
        ("langchain", "langgraph")
    )


def _type_checking_block(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
    )


def _import_time_imports(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Imports that run when the module is imported: module level, class bodies
    and module-level blocks, but not function bodies or TYPE_CHECKING blocks."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if _type_checking_block(node):
            continue
        if isinstance(node, ast.Import):
            yield from ((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module, node.lineno
        else:
            stack.extend(ast.iter_child_nodes(node))


def _modules(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_of_the_core_imports_upstream_or_a_vendor_at_import_time():
    offences = []
    for path in _modules(PACKAGE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, line in _import_time_imports(tree):
            if _forbidden(module):
                offences.append(f"{path.relative_to(REPO).as_posix()}:{line} imports {module}")
    assert not offences, "import inside the function that needs it:\n" + "\n".join(offences)


def test_importing_every_core_module_loads_neither_upstream_nor_a_vendor(tmp_path):
    code = f"""
import importlib, json, pkgutil, sys
import tradingagents_quant
skipped = []
for info in pkgutil.walk_packages(tradingagents_quant.__path__, "tradingagents_quant."):
    try:
        importlib.import_module(info.name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in {sorted(_OPTIONAL)!r}:
            skipped.append(info.name)
            continue
        raise
print(json.dumps({{"modules": sorted(sys.modules), "skipped": skipped}}))
"""
    # A working directory with nothing above it that could be a project .env.
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    loaded = sorted(m for m in report["modules"] if _forbidden(m))
    assert loaded == [], f"importing the core pulled in: {loaded}"


def test_upstream_reaches_the_core_only_through_the_declared_seam():
    reaching = set()
    for root in (REPO / "tradingagents", REPO / "cli"):
        for path in _modules(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n.split(".")[0] == "tradingagents_quant" for n in names):
                    reaching.add(path.relative_to(REPO).as_posix())
    assert reaching <= ALLOWED_UPSTREAM_SEAMS, sorted(reaching - ALLOWED_UPSTREAM_SEAMS)
    # The decision-quality sweep stays what its docstring says it is.
    assert "tradingagents/backtest.py" not in reaching
