"""Add retrieval metadata without changing existing chunk IDs or text.

The default output is isolated under ``data/stage1/chunks``. Production files
are not overwritten unless ``--in-place`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "chunks"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stage1" / "chunks"
CHUNK_FILES = ("ds_chunks.json", "os_chunks.json", "cn_chunks.json")

CHINESE_NUMBER = "一二三四五六七八九十百千万零〇两0-9"


def clean_document_title(source: str) -> str:
    title = Path(source).stem
    title = re.sub(r"\s*\(Z-Library\)(?:\(OCR\))?\s*$", "", title, flags=re.IGNORECASE)
    return title.strip()


def compact_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def detect_heading(line: str) -> Optional[Tuple[int, str]]:
    """Return ``(level, heading)`` for conservative chapter-like lines."""
    text = compact_line(line)
    if not text or len(text) > 140:
        return None

    if re.match(rf"^第\s*[{CHINESE_NUMBER}]+\s*[编篇部]\b", text):
        return 1, text
    if re.match(rf"^第\s*[{CHINESE_NUMBER}]+\s*章\b", text):
        return 1, text
    if re.match(rf"^第\s*[{CHINESE_NUMBER}]+\s*节\b", text):
        return 2, text
    if re.match(r"^(PART|Part)\s+[IVXLC0-9]+\b", text):
        return 1, text
    if re.match(r"^(CHAPTER|Chapter)\s+[A-Z0-9IVXLC]+\b", text):
        return 1, text

    numbered = re.match(r"^(\d+(?:\.\d+){1,4})[.、]?\s+(.+)$", text)
    if numbered:
        title = numbered.group(2).strip()
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", title))
        alpha_count = len(re.findall(r"[A-Za-z]", title))
        if (
            3 <= len(title) <= 100
            and max(cjk_count, alpha_count) >= 4
            and not re.search(r"[。！？!?]$", title)
            and not re.search(r"[=+×÷·]", title)
        ):
            depth = min(3, numbered.group(1).count(".") + 1)
            return depth, text

    top_level = re.match(r"^(\d{1,2})[.、]\s*(.+)$", text)
    if top_level:
        title = top_level.group(2).strip()
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", title))
        alpha_count = len(re.findall(r"[A-Za-z]", title))
        if (
            3 <= len(title) <= 100
            and max(cjk_count, alpha_count) >= 4
            and not re.search(r"[。！？.!?]$", title)
            and not re.search(r"[=+×÷·]", title)
        ):
            return 1, text
    return None


def headings_from_chunk(text: str) -> list[Tuple[int, str]]:
    # The existing chunker flushes before a detected heading, so a structural
    # heading should be the first non-empty line. Scanning deeper lines turns
    # numbered examples, equations and list items into false chapter paths.
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    heading = detect_heading(first_line)
    return [heading] if heading else []


def update_heading_state(state: Dict[int, str], headings: Iterable[Tuple[int, str]]) -> None:
    for level, title in headings:
        state[level] = title
        for deeper in [key for key in state if key > level]:
            del state[deeper]


def build_retrieval_context(chunk: Dict[str, Any]) -> str:
    fields = [
        ("课程", chunk.get("course")),
        ("文档", chunk.get("document_title")),
        ("章节", chunk.get("chapter_path")),
    ]
    return "；".join(f"{label}：{value}" for label, value in fields if value)


def enrich_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a JSON list: {input_path}")

    original_ids = [str(item["id"]) for item in chunks]
    original_text_hashes = {
        str(item["id"]): hashlib.sha256(str(item.get("text", "")).encode("utf-8")).hexdigest()
        for item in chunks
    }
    heading_state: Dict[str, Dict[int, str]] = defaultdict(dict)
    enriched: list[Dict[str, Any]] = []

    for item in chunks:
        result = dict(item)
        source = str(result.get("source") or "")
        state = heading_state[source]
        update_heading_state(state, headings_from_chunk(str(result.get("text") or "")))
        path_parts = [state[level] for level in sorted(state)]

        result["course"] = result.get("subject")
        result["document_title"] = clean_document_title(source)
        result["chapter_path"] = " > ".join(path_parts) if path_parts else None
        result["chapter_titles"] = path_parts
        result["page_start"] = result.get("page_start")
        result["page_end"] = result.get("page_end")
        result["document_type"] = result.get("document_type") or "textbook"
        result["language"] = result.get("lang")
        result["metadata_version"] = "stage1-v1"
        result["retrieval_context"] = build_retrieval_context(result)
        enriched.append(result)

    enriched_ids = [str(item["id"]) for item in enriched]
    if enriched_ids != original_ids:
        raise AssertionError(f"Chunk ID order changed for {input_path}")
    for item in enriched:
        text_hash = hashlib.sha256(str(item.get("text", "")).encode("utf-8")).hexdigest()
        if text_hash != original_text_hashes[str(item["id"])]:
            raise AssertionError(f"Chunk text changed: {item['id']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    chapter_count = sum(bool(item.get("chapter_path")) for item in enriched)
    source_counts = Counter(str(item.get("source")) for item in enriched)
    return {
        "file": input_path.name,
        "chunks": len(enriched),
        "chunks_with_chapter_path": chapter_count,
        "chapter_coverage": chapter_count / len(enriched) if enriched else 0.0,
        "sources": len(source_counts),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files. The default keeps production chunks untouched.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = input_dir if args.in_place else args.output_dir.resolve()
    reports = []
    for filename in CHUNK_FILES:
        input_path = input_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        reports.append(enrich_file(input_path, output_dir / filename))

    summary = {
        "metadata_version": "stage1-v1",
        "in_place": args.in_place,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": reports,
        "total_chunks": sum(item["chunks"] for item in reports),
        "chunk_ids_preserved": True,
        "chunk_text_preserved": True,
    }
    summary_path = output_dir.parent / "metadata_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for report in reports:
        print(
            f"{report['file']}: {report['chunks']} chunks | "
            f"chapter coverage={report['chapter_coverage']:.1%}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
