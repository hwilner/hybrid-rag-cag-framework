"""Shared test configuration.

Two things are set up here.

The repository root is put on ``sys.path`` so that the ``src.<module>`` import
style used by ``tests/test_loss_differentiable.py`` works.

And the torch-dependent suite is skipped when torch is not installed. The
import of ``src.trainable_losses`` in that file happens at collection time, so
without this the whole directory failed to collect in an environment without
torch -- which is one of the reasons the suite was never exercised in CI.
Skipping the file at collection time lets ``pytest tests/ -q`` work in both
environments: the light tests run everywhere, and the heavy ones run wherever
torch is available.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The modules in src/ are imported as top-level modules, the way
# `src/hybrid_rag_cag_system.py` and `src/run_honest_evaluation.py` import
# them. Putting src/ on the path here means the test session shares one
# module object with the code under test, so that an exception class raised by
# `generation_metrics` is the same class the tests catch.
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Test modules that need torch and transformers at import time.
HEAVY_MODULES = ["test_loss_differentiable.py"]

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    # `collect_ignore` is the supported way to skip a module at collection
    # time, without the module ever being imported.
    collect_ignore = HEAVY_MODULES
