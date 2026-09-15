"""Embedding model registry shared by indexing and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize_embeddings: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SPECS: dict[str, EmbeddingSpec] = {
    "minilm": EmbeddingSpec(
        key="minilm",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "e5-base": EmbeddingSpec(
        key="e5-base",
        model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "bge-m3": EmbeddingSpec(
        key="bge-m3",
        model_name="BAAI/bge-m3",
    ),
}


def get_embedding_spec(key_or_name: str) -> EmbeddingSpec:
    normalized = key_or_name.strip()
    if normalized in MODEL_SPECS:
        return MODEL_SPECS[normalized]
    for spec in MODEL_SPECS.values():
        if spec.model_name == normalized:
            return spec
    supported = ", ".join(sorted(MODEL_SPECS))
    raise KeyError(f"Unknown embedding model {key_or_name!r}; supported keys: {supported}")


def spec_from_config(config: Mapping[str, Any]) -> EmbeddingSpec:
    model_key = str(config.get("model_key") or "").strip()
    model_name = str(config.get("model_name") or "").strip()
    if model_key:
        base = get_embedding_spec(model_key)
    elif model_name:
        base = get_embedding_spec(model_name)
    else:
        base = MODEL_SPECS["minilm"]
    return EmbeddingSpec(
        key=base.key,
        model_name=model_name or base.model_name,
        query_prefix=str(config.get("query_prefix", base.query_prefix)),
        passage_prefix=str(config.get("passage_prefix", base.passage_prefix)),
        normalize_embeddings=bool(config.get("normalize_embeddings", config.get("use_cosine", True))),
    )


def compose_passage_text(chunk: Mapping[str, Any], text_mode: str) -> str:
    text = str(chunk.get("text") or "").strip()
    if text_mode == "raw":
        return text
    if text_mode != "contextual":
        raise ValueError(f"Unsupported text mode: {text_mode}")
    context = str(chunk.get("retrieval_context") or "").strip()
    return f"{context}\n正文：{text}" if context else text
