"""AST-based dependency boundary enforcement (§5.2).

This scans source files' import statements (never executes them) and checks
each module against a table of forbidden imports. As new modules are added
in later phases, add their rules to `FORBIDDEN_IMPORTS` below rather than
writing a new ad-hoc test.

Phase 1 note: most of the modules this test protects (marl.algorithms,
marl.networks, backend.*) don't exist yet. The rules are declared now so
that later phases cannot forget to add the check when the files appear —
the rule for a not-yet-existing module is simply skipped (see
`_iter_existing_targets`).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# module_path (dotted, matching its file under REPO_ROOT) -> list of
# forbidden import prefixes for that module.
FORBIDDEN_IMPORTS: dict[str, list[str]] = {
    "marl.envs.core": ["torch", "marl.algorithms", "marl.networks", "backend"],
    "marl.envs.pettingzoo_env": ["marl.algorithms"],
    "marl.envs.vector": ["marl.algorithms"],
    "marl.networks.actor": ["marl.envs.core.global_state", "marl.networks.critics"],
    "marl.algorithms": ["backend"],
    # §5.2: evaluation/analysis/replay never reach into training internals or
    # the study orchestration layer above them (Phase 6).
    "marl.evaluation": ["backend", "marl.training", "marl.experiments"],
    "marl.analysis": ["backend", "marl.training", "marl.experiments"],
    "marl.replay": ["backend", "marl.training", "marl.experiments"],
    "marl.experiments": ["backend"],
    "backend": [],  # backend may import marl.* freely; see policy.py rule below
}

# Files, not just packages, that need import-level scanning for actor-only
# purity (§12.4 test_actor_import_boundary).
FORBIDDEN_IMPORTS_BY_FILE: dict[str, list[str]] = {
    "marl/networks/actor.py": ["global_state", "critics"],
    "marl/algorithms/common/policy.py": ["global_state", "critics"],
}


def _module_to_path(dotted: str) -> Path | None:
    """Resolve a dotted module/package name to a file or package dir, if it exists."""
    parts = dotted.split(".")
    as_module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = REPO_ROOT.joinpath(*parts, "__init__.py")
    if as_package.is_file():
        return as_package.parent
    return None


def _iter_python_files(root: Path):
    if root.is_file():
        yield root
        return
    for f in root.rglob("*.py"):
        if "__pycache__" not in f.parts:
            yield f


def _imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_forbidden_imports_by_module():
    violations = []
    for dotted, forbidden in FORBIDDEN_IMPORTS.items():
        if not forbidden:
            continue
        target = _module_to_path(dotted)
        if target is None:
            continue  # module doesn't exist yet in this phase
        for f in _iter_python_files(target):
            for imp in _imports_in_file(f):
                for bad in forbidden:
                    if imp == bad or imp.startswith(bad + "."):
                        violations.append(f"{f.relative_to(REPO_ROOT)} imports forbidden '{imp}'")
    assert not violations, "\n".join(violations)


def test_forbidden_imports_by_file():
    violations = []
    for rel_path, forbidden_substrings in FORBIDDEN_IMPORTS_BY_FILE.items():
        f = REPO_ROOT / rel_path
        if not f.is_file():
            continue  # not implemented yet in this phase
        for imp in _imports_in_file(f):
            for bad in forbidden_substrings:
                if bad in imp:
                    violations.append(f"{rel_path} imports forbidden '{imp}'")
    assert not violations, "\n".join(violations)


def test_no_global_numpy_random_in_env_core():
    """Environment code must use its own np.random.Generator, never the
    numpy global RNG (`np.random.<fn>` other than `default_rng`/`Generator`/
    `SeedSequence`), per §34."""
    target = REPO_ROOT / "marl" / "envs" / "core"
    if not target.is_dir():
        return
    allowed_np_random_attrs = {"default_rng", "Generator", "SeedSequence"}
    violations = []
    for f in _iter_python_files(target):
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "np"
                and node.value.attr == "random"
                and node.attr not in allowed_np_random_attrs
            ):
                violations.append(f"{f.relative_to(REPO_ROOT)}: np.random.{node.attr}")
    assert not violations, "\n".join(violations)


def test_no_source_file_exceeds_soft_line_limit():
    """Soft limit per §5.2: no source file should exceed ~400 lines without
    justification. This test warns via assertion message but only fails
    hard past 600 lines, to allow occasional justified exceptions to be
    caught by human review rather than blocking CI outright."""
    hard_limit = 600
    violations = []
    for pkg in ("marl", "backend"):
        root = REPO_ROOT / pkg
        if not root.is_dir():
            continue
        for f in _iter_python_files(root):
            n_lines = sum(1 for _ in f.open())
            if n_lines > hard_limit:
                violations.append(f"{f.relative_to(REPO_ROOT)}: {n_lines} lines")
    assert not violations, "\n".join(violations)
