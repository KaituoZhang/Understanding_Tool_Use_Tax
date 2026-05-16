"""
Compatibility wrapper for the gate-enabled mechanism ablation entrypoint.

The public release keeps the original filename so paper references still map
cleanly, while the actual implementation now lives in
`run_mechanism_ablation.py`.
"""

from .run_mechanism_ablation import main, run_mechanism_ablation


__all__ = ["main", "run_mechanism_ablation"]


if __name__ == "__main__":
    main()
