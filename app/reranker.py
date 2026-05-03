from typing import List, Dict
from sentence_transformers import CrossEncoder


class BGECrossEncoderReranker:
    """
    Cross-Encoder reranker:
    input: (query, passage) -> score
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        candidates: retriever 返回的 list[dict], 每个 dict 至少包含 "text" 和 "score"
        返回：按 rerank_score 降序排序后的 top_k
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
