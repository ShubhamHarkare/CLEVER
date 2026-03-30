"""
Experiment run manifest generator.

Responsible for capturing deterministic provenance metadata about
an execution environment (hardware, exact package versions, Git SHA)
so that every JSON output file represents a theoretically perfectly 
reproducible artifact for the publication.
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)


def _get_git_info() -> dict:
    """Safely extract git commit SHA and dirty state."""
    info = {"commit": "unknown", "branch": "unknown", "is_dirty": False}
    try:
        # Check if inside a git repo
        subprocess.check_call(
            ["git", "rev-parse"], 
            stderr=subprocess.DEVNULL, 
            stdout=subprocess.DEVNULL
        )
        
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
        
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8").strip()
        
        # Check dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"]
        ).decode("utf-8").strip()
        info["is_dirty"] = len(status) > 0
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Git info not available (not a repo or git not in PATH).")
        
    return info


def _get_package_versions() -> dict:
    """Capture precise versions of ML/Data tools."""
    versions = {}
    
    # Required core
    import numpy
    import pandas
    
    versions["numpy"] = numpy.__version__
    versions["pandas"] = pandas.__version__
    
    # PyTorch (optional — may not be installed on CPU-only nodes)
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not_installed"
    
    # Optional FAISS (may be cpu or gpu)
    try:
        import faiss
        versions["faiss"] = faiss.__version__
    except ImportError:
        versions["faiss"] = "not_installed"
        
    # Optional HF tools
    try:
        import transformers
        import sentence_transformers
        versions["transformers"] = transformers.__version__
        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
        
    return versions


def _get_hardware_info() -> dict:
    """Capture critical hardware specs."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }


def generate_manifest(config: dict) -> dict:
    """
    Generate a full publication-grade provenance manifest.
    
    Args:
        config: The parsed configuration dictionary for the experiment run.
        
    Returns:
        Dict representing all the metadata for the run.
    """
    # Create configuration hash to catch silent config edits later
    config_json_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_json_bytes).hexdigest()[:12]
    
    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "command_line": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "hardware": _get_hardware_info(),
        "packages": _get_package_versions(),
        "git": _get_git_info(),
        "config_hash": config_hash,
    }
    
    return manifest
