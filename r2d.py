"""Backward-compatible R2D command-line launcher.

The implementation lives in :mod:`r2d_simulation`; this file is kept so the
historical ``python r2d.py`` command continues to work.
"""

from r2d_simulation.workflow import run


def main(*args, **kwargs):
    """Backward-compatible programmatic simulation entry point."""
    return run(*args, **kwargs)


if __name__ == "__main__":
    from r2d_simulation.main import main as cli_main

    cli_main()


__all__ = ["main", "run"]
