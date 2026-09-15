"""Production context assembly policy for Stage-2 Parent-Child retrieval."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


COMPARISON_MARKERS = ("区别", "比较", "对比", "差异", "异同", "优缺点")
ENUMERATION_MARKERS = (
    "哪些", "几种", "两种", "三种", "四种", "五种", "六种", "七种", "八种",
    "九种", "十种", "枚举", "列出", "包括",
)


def infer_query_type(query: str) -> str:
    """Infer the small routing intent used by the validated context policy."""
    text = query.strip()
    if any(marker in text for marker in COMPARISON_MARKERS) or ("和" in text and "特性" in text):
        return "comparison"
    if any(marker in text for marker in ENUMERATION_MARKERS):
        return "enumeration"
    if any(f"{number}个" in text for number in "一二两三四五六七八九十123456789"):
        return "enumeration"
    if any(marker in text for marker in ("什么", "定义", "解释")):
        return "definition"
    return "enumeration"


def insert_baseline_rescue(
    parent_results: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
    parent_store: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Insert the first unique legacy BGE Parent at its original rank."""
    selected = [dict(item) for item in parent_results]
    selected_ids = {str(item["id"]) for item in selected}
    for baseline_rank, baseline in enumerate(baseline_results):
        doc_id = str(baseline["id"])
        if doc_id in selected_ids or doc_id not in parent_store:
            continue
        rescue = dict(parent_store[doc_id])
        rescue.update({
            "id": doc_id,
            "parent_id": doc_id,
            "parent_score": float(
                baseline.get("rerank_score", baseline.get("score", baseline.get("rrf_score", 0.0)))
            ),
            "baseline_rescue_rank": baseline_rank + 1,
            "context_role": "baseline_rescue",
        })
        selected.insert(min(baseline_rank, len(selected)), rescue)
        return selected, doc_id
    return selected, None


def _main_score(item: Mapping[str, Any]) -> float:
    for key in ("parent_score", "parent_rerank_score", "rerank_score", "score", "rrf_score"):
        if item.get(key) is not None:
            return float(item[key])
    return 0.0


def build_parent_context(
    parents: Sequence[Mapping[str, Any]],
    *,
    max_chars: int,
    highlights: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build cited Parent context, optionally foregrounding one Child per Parent."""
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    highlights = highlights or {}
    for rank, original in enumerate(parents, 1):
        parent = dict(original)
        parent_id = str(parent.get("parent_id") or parent.get("id") or "")
        chosen = list(highlights.get(parent_id, ()))
        course = parent.get("course") or parent.get("subject")
        title = parent.get("document_title") or parent.get("source")
        chapter = parent.get("chapter_path")
        score = _main_score(parent)
        header = (
            f"[{rank}] subject={parent.get('subject')} | course={course} | "
            f"document={title} | chapter={chapter} | id={parent_id} | score={score:.4f}\n"
        )
        if chosen:
            excerpt = "\n\n".join(str(child.get("text") or "") for child in chosen)
            body = f"关键命中片段：\n{excerpt}\n\n所属 Parent 完整上下文：\n{parent.get('text', '')}"
            parent["highlight_child_ids"] = [str(child.get("id")) for child in chosen]
        else:
            body = str(parent.get("text") or "")
            parent["highlight_child_ids"] = []
        blocks.append(header + body)
        parent["rank"] = rank
        parent["score"] = score
        parent.setdefault(
            "rrf_score", parent.get("child_aggregate_score", parent.get("parent_score", score))
        )
        parent.setdefault(
            "rerank_score", parent.get("parent_rerank_score", parent.get("parent_score", score))
        )
        retrieval.append(parent)
        citations.append({
            "rank": rank, "score": score, "subject": parent.get("subject"),
            "course": course, "document_title": title, "chapter_path": chapter,
            "page": parent.get("page") or parent.get("page_number"),
            "chunk_file": parent.get("chunk_file"), "id": parent_id,
            "highlight_child_ids": parent["highlight_child_ids"],
        })
    context = "\n\n---\n\n".join(blocks)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n...[context truncated]"
    return context, citations, retrieval
