"""CLI entry points for megalodon-enwik8-jax."""

from __future__ import annotations


def train_main():
    """Entry point for train-megalodon CLI."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import and run train.py main
    import runpy

    runpy.run_path(str(project_root / "train.py"), run_name="__main__")


def inference_main():
    """Entry point for infer-megalodon CLI."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Import and run inference.py main
    import runpy

    runpy.run_path(str(project_root / "inference.py"), run_name="__main__")
