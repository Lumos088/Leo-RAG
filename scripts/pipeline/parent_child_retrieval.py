"""Experimental Child retrieval and Parent aggregation for Stage 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_PATH = PROJECT_ROOT / "config/parent_child.json"
from scripts.pipeline.corpus_files import discover_chunk_files
from scripts.pipeline.query_router import detect_subject


def load_parent_child_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "dense_k",
        "bm25_k",
        "rrf_k",
        "rerank_candidates",
        "parent_top_k",
        "context_token_budget",
    ):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("dense_weight", "bm25_weight"):
        config[key] = float(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    aggregation = str(config.get("parent_score_aggregation", "max"))
    if aggregation not in {"max", "top2_mean", "max_rank_bonus"}:
        raise ValueError(
            "parent_score_aggregation must be one of: max, top2_mean, max_rank_bonus"
        )
    config["parent_score_aggregation"] = aggregation
    config["parent_rank_bonus_weight"] = float(config.get("parent_rank_bonus_weight", 0.05))
    if config["parent_rank_bonus_weight"] < 0:
        raise ValueError("parent_rank_bonus_weight must be non-negative")
    config["parent_rerank_enabled"] = bool(config.get("parent_rerank_enabled", False))
    config["parent_rerank_candidates"] = int(config.get("parent_rerank_candidates", 20))
    if config["parent_rerank_candidates"] <= 0:
        raise ValueError("parent_rerank_candidates must be positive")
    config["parent_rerank_mode"] = str(config.get("parent_rerank_mode", "replace"))
    if config["parent_rerank_mode"] not in {"replace", "rrf"}:
        raise ValueError("parent_rerank_mode must be replace or rrf")
    config["parent_rrf_k"] = int(config.get("parent_rrf_k", 20))
    if config["parent_rrf_k"] <= 0:
        raise ValueError("parent_rrf_k must be positive")
    for key in ("parent_child_rank_weight", "parent_bge_rank_weight"):
        config[key] = float(config.get(key, 1.0))
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    config["baseline_rescue_enabled"] = bool(config.get("baseline_rescue_enabled", True))
    config["definition_child_highlight_enabled"] = bool(
        config.get("definition_child_highlight_enabled", True)
    )
    config["child_highlights_per_parent"] = int(config.get("child_highlights_per_parent", 1))
    if config["child_highlights_per_parent"] <= 0:
        raise ValueError("child_highlights_per_parent must be positive")
    config["max_context_chars"] = int(config.get("max_context_chars", 16000))
    if config["max_context_chars"] <= 0:
        raise ValueError("max_context_chars must be positive")
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_parent_store(parent_dir: Path) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for filename in discover_chunk_files(parent_dir):
        path = parent_dir / filename
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            parent_id = str(record.get("parent_id") or record.get("id") or "")
            if not parent_id:
                raise ValueError(f"Parent without ID: {path}")
            if parent_id in parents:
                raise ValueError(f"Duplicate parent ID: {parent_id}")
            parents[parent_id] = record
    return parents


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for channel, results, weight in (
        ("dense", dense_results, config["dense_weight"]),
        ("bm25", bm25_results, config["bm25_weight"]),
    ):
        for rank, result in enumerate(results, 1):
            child_id = str(result["id"])
            item = merged.setdefault(child_id, dict(result))
            item[f"{channel}_rank"] = rank
            item[f"{channel}_score"] = float(
                result.get("bm25_score", result.get("score", 0.0))
                if channel == "bm25"
                else result.get("score", 0.0)
            )
            item["rrf_score"] = item.get("rrf_score", 0.0) + weight / (
                config["rrf_k"] + rank
            )
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)


def aggregate_parents(
    ranked_children: list[dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    parent_top_k: int,
    aggregation: str = "max",
    rank_bonus_weight: float = 0.05,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for child_rank, child in enumerate(ranked_children, 1):
        parent_id = str(child.get("parent_id") or "")
        if parent_id not in parents:
            raise KeyError(f"Child points to unknown parent: {child.get('id')}")
        child_score = float(
            child.get("rerank_score")
            if child.get("rerank_score") is not None
            else child.get("rrf_score")
            if child.get("rrf_score") is not None
            else child.get("score", 0.0)
        )
        if parent_id not in aggregated:
            parent = dict(parents[parent_id])
            parent.update(
                {
                    "id": parent_id,
                    "parent_id": parent_id,
                    "parent_score": child_score,
                    "best_child_rank": child_rank,
                    "matched_child_ids": [],
                    "matched_children": [],
                }
            )
            aggregated[parent_id] = parent
        parent = aggregated[parent_id]
        parent["parent_score"] = max(float(parent["parent_score"]), child_score)
        parent["best_child_rank"] = min(int(parent["best_child_rank"]), child_rank)
        parent["matched_child_ids"].append(str(child["id"]))
        parent["matched_children"].append(
            {
                "child_id": str(child["id"]),
                "child_rank": child_rank,
                "score": child_score,
                "source_char_start": child.get("source_char_start"),
                "source_char_end": child.get("source_char_end"),
                "dense_rank": child.get("dense_rank"),
                "bm25_rank": child.get("bm25_rank"),
                "rrf_score": child.get("rrf_score"),
                "rerank_score": child.get("rerank_score"),
            }
        )
    for parent in aggregated.values():
        child_scores = sorted(
            (float(item["score"]) for item in parent["matched_children"]), reverse=True
        )
        max_score = child_scores[0]
        if aggregation == "max":
            parent_score = max_score
        elif aggregation == "top2_mean":
            selected = child_scores[:2]
            parent_score = sum(selected) / len(selected)
        elif aggregation == "max_rank_bonus":
            parent_score = max_score + rank_bonus_weight / max(1, int(parent["best_child_rank"]))
        else:
            raise ValueError(f"Unsupported parent aggregation: {aggregation}")
        parent["parent_score"] = parent_score
        parent["parent_score_aggregation"] = aggregation
        parent["max_child_score"] = max_score
        parent["top2_child_mean"] = sum(child_scores[:2]) / len(child_scores[:2])
    ordered = sorted(
        aggregated.values(),
        key=lambda item: (-float(item["parent_score"]), int(item["best_child_rank"])),
    )
    return ordered[:parent_top_k]


def rerank_parents(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: Any,
    top_k: int,
    mode: str = "replace",
    rrf_k: int = 20,
    child_rank_weight: float = 1.0,
    bge_rank_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Rerank Parent candidates without replacing their original context text."""
    inputs = []
    by_id = {str(parent["id"]): parent for parent in candidates}
    for parent in candidates:
        context = str(parent.get("retrieval_context") or "").strip()
        text = str(parent.get("text") or "").strip()
        passage = f"{context}\n{text}" if context else text
        inputs.append({"id": str(parent["id"]), "text": passage, "score": parent["parent_score"]})
    child_rank_by_id = {str(parent["id"]): rank for rank, parent in enumerate(candidates, 1)}
    ranked = reranker.rerank(query, inputs, top_k=len(inputs))
    output = []
    for bge_rank, item in enumerate(ranked, 1):
        parent = by_id[str(item["id"])]
        parent["child_aggregate_score"] = float(parent["parent_score"])
        parent["parent_rerank_score"] = float(item["rerank_score"])
        parent["child_aggregate_rank"] = child_rank_by_id[str(item["id"])]
        parent["parent_rerank_rank"] = bge_rank
        if mode == "replace":
            parent["parent_score"] = float(item["rerank_score"])
        elif mode == "rrf":
            parent["parent_score"] = (
                child_rank_weight / (rrf_k + parent["child_aggregate_rank"])
                + bge_rank_weight / (rrf_k + bge_rank)
            )
        else:
            raise ValueError(f"Unsupported parent rerank mode: {mode}")
        parent["parent_rerank_mode"] = mode
        output.append(parent)
    if mode == "rrf":
        output.sort(
            key=lambda item: (
                -float(item["parent_score"]),
                int(item["parent_rerank_rank"]),
                int(item["child_aggregate_rank"]),
            )
        )
    return output[:top_k]


class ParentChildRetriever:
    def __init__(
        self,
        *,
        config_path: str | Path = CONFIG_PATH,
        device: str | None = None,
        load_reranker: bool = True,
        reranker: Any | None = None,
    ) -> None:
        from app.bm25_retriever import BM25Retriever
        from scripts.pipeline.retriever import FaissRetriever

        self.config = load_parent_child_config(Path(config_path).resolve())
        self.child_index_dir = resolve_project_path(self.config["child_index_dir"])
        self.parent_dir = resolve_project_path(self.config["parent_dir"])
        self.child_retriever = FaissRetriever(index_dir=self.child_index_dir, device=device)
        self.children = self.child_retriever.metadatas
        self.children_by_parent: dict[str, list[dict[str, Any]]] = {}
        for child in self.children:
            parent_id = str(child.get("parent_id") or "")
            self.children_by_parent.setdefault(parent_id, []).append(child)
        self.bm25 = BM25Retriever(self.children)
        self.bm25_by_subject = {
            subject: BM25Retriever(
                [item for item in self.children if item.get("subject") == subject]
            )
            for subject in {
                str(item.get("subject"))
                for item in self.children
                if item.get("subject")
            }
        }
        self.parents = load_parent_store(self.parent_dir)
        self.reranker = reranker
        if load_reranker and self.reranker is None:
            from app.reranker import BGECrossEncoderReranker

            self.reranker = BGECrossEncoderReranker()

    def select_parent_highlights(
        self,
        query: str,
        parent_ids: list[str],
        *,
        per_parent: int = 1,
        use_rerank: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """Select the strongest Child evidence inside each chosen Parent."""
        if per_parent <= 0:
            raise ValueError("per_parent must be positive")
        candidates: list[dict[str, Any]] = []
        for parent_id in parent_ids:
            for child in self.children_by_parent.get(parent_id, []):
                item = dict(child)
                item.setdefault("score", 0.0)
                candidates.append(item)
        if use_rerank:
            if self.reranker is None:
                raise RuntimeError("Child highlight reranking requested without a reranker")
            candidates = self.reranker.rerank(query, candidates, top_k=len(candidates))
        selected: dict[str, list[dict[str, Any]]] = {}
        for child in candidates:
            parent_id = str(child.get("parent_id") or "")
            bucket = selected.setdefault(parent_id, [])
            if len(bucket) < per_parent:
                bucket.append(child)
        return selected

    def search(
        self,
        query: str,
        *,
        parent_top_k: int | None = None,
        use_bm25: bool = True,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        parent_top_k = parent_top_k or self.config["parent_top_k"]
        subject = detect_subject(query)
        dense_kwargs = {"top_k": self.config["dense_k"]}
        if subject:
            dense_kwargs["subject"] = subject
        dense = self.child_retriever.search(query, **dense_kwargs)
        bm25: list[dict[str, Any]] = []
        if use_bm25:
            bm25_engine = getattr(self, "bm25_by_subject", {}).get(subject, self.bm25)
            bm25 = bm25_engine.search(query, top_k=self.config["bm25_k"])
            candidates = rrf_fuse(dense, bm25, self.config)
        else:
            candidates = [dict(item) for item in dense]

        candidates = candidates[: self.config["rerank_candidates"]]
        if use_rerank:
            if self.reranker is None:
                raise RuntimeError("Reranking requested but reranker was not loaded")
            ranked_children = self.reranker.rerank(
                query, candidates, top_k=len(candidates)
            )
        else:
            ranked_children = candidates
        parent_candidate_k = parent_top_k
        if self.config["parent_rerank_enabled"] and use_rerank:
            parent_candidate_k = max(parent_top_k, self.config["parent_rerank_candidates"])
        parents = aggregate_parents(
            ranked_children,
            self.parents,
            parent_candidate_k,
            aggregation=self.config["parent_score_aggregation"],
            rank_bonus_weight=self.config["parent_rank_bonus_weight"],
        )
        if self.config["parent_rerank_enabled"] and use_rerank:
            if self.reranker is None:
                raise RuntimeError("Parent reranking requested but reranker was not loaded")
            parents = rerank_parents(
                query,
                parents,
                self.reranker,
                parent_top_k,
                mode=self.config["parent_rerank_mode"],
                rrf_k=self.config["parent_rrf_k"],
                child_rank_weight=self.config["parent_child_rank_weight"],
                bge_rank_weight=self.config["parent_bge_rank_weight"],
            )
        return {
            "query": query,
            "variant": self.config["variant"],
            "parents": parents,
            "ranked_children": ranked_children,
            "dense_children": dense,
            "bm25_children": bm25,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--device")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    engine = ParentChildRetriever(device=args.device, load_reranker=not args.no_rerank)
    result = engine.search(args.query, use_rerank=not args.no_rerank)
    for rank, parent in enumerate(result["parents"], 1):
        print(
            f"{rank}. {parent['id']} score={parent['parent_score']:.4f} "
            f"children={parent['matched_child_ids']}"
        )
