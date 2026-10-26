"""Locate and load the Tier-2 benchmark for the checkout being evaluated.

Why this module exists
----------------------
Three scripts in this repository need the same benchmark: the 100-document,
55-question Tier-2 set. They are ``src/real_ablation.py``,
``src/ceiling_test.py`` and ``src/diagnose_failures.py``.

Each of them used to carry its own copy of a loader that reached into
``src/option3_full_scale_evaluation.py`` and ``exec``-ed the
``create_large_scale_dataset`` function out of it with :func:`ast`.

That file was quarantined under ``retracted/`` on 2026-10-13, because the
numbers it produced were fabricated (see ``docs/INTEGRITY.md``). The dataset
itself was never the problem -- it is a pure function of two literal lists --
so it was extracted verbatim into ``src/tier2_dataset.py``. The three loaders
were never updated to follow it, so all three died with::

    FileNotFoundError: .../src/option3_full_scale_evaluation.py

before doing any work at all. The three commands the README documents under
"Reproducing everything" could not run.

This module is the single place that knows where the benchmark lives. It
prefers the canonical module, falls back to the legacy location for branches
that predate the extraction, and reports a clear error when neither is
present.

The fallback still parses the legacy file rather than importing it, because the
legacy file's module scope has side effects (it downloads models and writes
result files at import time). Only the one dataset function is extracted, and
it is called in a namespace that contains just the names that function uses.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

# The module that holds the dataset in this checkout, relative to ``src/``.
CANONICAL_MODULE = "tier2_dataset.py"

# The pre-extraction location. Still needed for branches that predate the
# 2026-10-13 extraction; see the module docstring for why it is parsed rather
# than imported.
LEGACY_MODULE = "option3_full_scale_evaluation.py"


def _import_module_from_path(path: str, name: str) -> ModuleType:
    """Import the module at ``path`` under the name ``name``.

    Loading by path rather than by name is deliberate: the caller may be
    evaluating a checkout other than this one, and the two must not be
    confused for one another in ``sys.modules``.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load a module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_dataset_function(path: str):
    """Return ``create_large_scale_dataset`` extracted from a legacy file.

    The legacy module cannot simply be imported: its module scope downloads
    models and writes result files. Only the dataset function is extracted,
    and it is executed in a namespace holding just the names it uses.
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_large_scale_dataset":
            import collections
            import datetime
            import logging

            import numpy as np

            namespace: Dict[str, Any] = {
                "np": np,
                "logger": logging.getLogger("dataset_loader"),
                "defaultdict": collections.defaultdict,
                "Counter": collections.Counter,
                "datetime": datetime.datetime,
            }
            exec(  # noqa: S102 - restricted to one known function definition.
                compile(ast.Module(body=[node], type_ignores=[]), path, "exec"),
                namespace,
            )
            return namespace["create_large_scale_dataset"]

    raise ImportError(
        f"{path!r} does not define create_large_scale_dataset(); it is not a "
        "Tier-2 dataset module"
    )


def load_tier2_dataset(
    branch_dir: str,
    fallback_dir: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """Return ``(corpus, questions)`` for the Tier-2 benchmark.

    Args:
        branch_dir: the checkout whose dataset should be loaded.
        fallback_dir: a checkout to fall back on when ``branch_dir`` has
            neither the canonical nor the legacy dataset module. Defaults to
            the directory containing this file.

    Returns:
        The corpus and the question list, as produced by
        ``create_large_scale_dataset()``.

    Raises:
        FileNotFoundError: if no dataset module could be found. The message
            names every path that was tried, because this failure used to
            present as a bare ``FileNotFoundError`` on a path the reader would
            not expect.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    fallback_dir = fallback_dir or os.path.dirname(here)

    tried = []
    # Each root is tried in full before the next one, so that the checkout
    # under evaluation wins over the fallback: a branch that still has the
    # dataset in the pre-extraction file must be measured on its own data.
    for root in (branch_dir, fallback_dir):
        canonical = os.path.join(root, "src", CANONICAL_MODULE)
        tried.append(canonical)
        if os.path.exists(canonical):
            module = _import_module_from_path(canonical, "tier2_dataset")
            return module.create_large_scale_dataset()

        legacy = os.path.join(root, "src", LEGACY_MODULE)
        tried.append(legacy)
        if os.path.exists(legacy):
            return _legacy_dataset_function(legacy)()

    raise FileNotFoundError(
        "could not find the Tier-2 dataset module. Tried:\n  "
        + "\n  ".join(tried)
        + "\nExpected one of them to contain create_large_scale_dataset()."
    )
