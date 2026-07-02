"""Cross-process file locks for shared mutable state.

The export driver runs several per-problem pipelines concurrently, each as its
own subprocess (``rma solve`` / ``rma push``), so plain threading locks cannot
protect the handful of files that are shared ACROSS problems:

  * documents/questions/_system_/literature (read-modify-write JSON index)
  * webapp/insights/datasets/<slug>.json and system.json (whole-file rewrite)
  * webapp/push_forward_state.json (job registry)
  * documents/discussions/index.tex (whole-file rewrite)
  * documents/pdf/master_<dataset>.pdf (+ latexmk aux files)
  * documents/strategy_memory.jsonl (append)

These are guarded with ``fcntl.flock`` on lock files under ``<repo>/.locks/``.
Per-problem artifacts (outputs/<ds>/<pid>, issues/<ds>/<pid>, per-question
insights, report_<pid> PDFs) are path-disjoint and need no locking.
"""

from __future__ import annotations

import fcntl
import functools
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(repo_root: Path, name: str):
    """Exclusive cross-process lock scoped to ``<repo_root>/.locks/<name>.lock``."""
    lock_dir = Path(repo_root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = open(lock_dir / f"{name}.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def locked(name: str):
    """Decorator for functions whose FIRST positional arg is ``repo_root``:
    the whole call runs under ``file_lock(repo_root, name)``."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(repo_root, *args, **kwargs):
            with file_lock(repo_root, name):
                return fn(repo_root, *args, **kwargs)
        return wrapper
    return decorator
