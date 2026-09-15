from rank_bm25 import BM25Okapi
import jieba


class BM25Retriever:
    def __init__(self, docs):
        self.tokenized_docs = [
            list(jieba.cut(d.get("retrieval_text") or d["text"])) for d in docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.docs = docs

    def search(self, query, top_k=10):
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for doc, score in ranked[:top_k]:
            doc = doc.copy()
            doc["score"] = float(score)
            doc["bm25_score"] = float(score)
            results.append(doc)
        return results
