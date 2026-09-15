"""Shared discovery rules for corpus JSON files.

The original project assumed exactly three subjects.  Stage 4 keeps those
files in their historical order, then appends any additional ``*_chunks.json``
files deterministically.
"""

from __future__ import annotations

from pathlib import Path


LEGACY_CHUNK_FILES = ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")


def discover_chunk_files(directory: Path) -> tuple[str, ...]:
    directory = Path(directory)
    present = {path.name for path in directory.glob("*_chunks.json") if path.is_file()}
    ordered = [name for name in LEGACY_CHUNK_FILES if name in present]
    ordered.extend(sorted(present - set(ordered)))
    if not ordered:
        raise FileNotFoundError(f"No *_chunks.json files found under {directory}")
    return tuple(ordered)
