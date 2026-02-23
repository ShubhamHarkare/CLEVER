"""
Environment validation utility.

Ensures that scripts are run with the officially supported Python
runtime to maintain reproducibility and avoid subtle library behavior
shifts across versions.
"""

import logging
import sys

logger = logging.getLogger(__name__)

SUPPORTED_MAJOR = 3
SUPPORTED_MINOR = 11

def require_supported_runtime() -> None:
    """
    Assert that the current Python interpreter matches the pinned version.
    Current target: Python 3.11.x
    
    Raises:
        RuntimeError: If the Python version is unsupported.
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    
    if major != SUPPORTED_MAJOR or minor != SUPPORTED_MINOR:
        msg = (
            f"UNSUPPORTED RUNTIME: CLEVER requires Python "
            f"{SUPPORTED_MAJOR}.{SUPPORTED_MINOR}.x for reproducible "
            f"execution and stable ML integrations. "
            f"Found: Python {major}.{minor}.{sys.version_info.micro}."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    
    logger.debug(f"Environment check passed: Python {major}.{minor}.{sys.version_info.micro}")

def pin_numpy_threads() -> None:
    """
    Pin common scientific computing libraries to a single thread
    to prevent oversubscription on shared compute nodes.
    Useful for standalone benchmarking where parallelism is managed manually.
    """
    import os
    
    thread_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS"
    ]
    for var in thread_vars:
        if var not in os.environ:
            os.environ[var] = "1"
            
    logger.debug("Pinned numpy computation threads to 1")
