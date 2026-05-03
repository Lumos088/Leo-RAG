"""
RAG Core Service
Extracted from app/rag_cli.py for Web UI usage
"""
import os
import sys
import json
import re
from typing import List, Dict, Generator, Optional
from openai import OpenAI, AsyncOpenAI

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
stage2_dir = os.path.dirname(parent_dir)
sys.path.append(stage2_dir)

from scripts.pipeline.retriever import FaissRetriever
from app.reranker import BGECrossEncoderReranker
from app.bm25_retriever import BM25Retriever

# Configuration
RECALL_N = 50
FUSION_K = 20
TOP_K = 5
MAX_CONTEXT_CHARS = 8000
VECTOR_DIR = os.path.join(stage2_dir, "vector_db")
META_PATH = os.path.join(VECTOR_DIR, "kb_meta.json")

MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def detect_lang(text: str) -> str:
    """Detect language: 'zh' for Chinese, 'en' for others"""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def get_main_score(r: Dict) -> float:
    """Get main score from result dict"""
    return (
        r.get("rerank_score")
        if r.get("rerank_score") is not None else
        r.get("score")
        if r.get("score") is not None else
        r.get("bm25_score")
        if r.get("bm25_score") is not None else
        r.get("rrf_score", 0.0)
    )


def rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion"""
    return 1.0 / (k + rank)


def build_context(results: List[Dict]) -> tuple[str, List[Dict]]:
    """Build context string and citations from results"""
    blocks = []
    citations = []

    for i, r in enumerate(results, 1):
        main_score = get_main_score(r)
        citations.append({
            "rank": i,
            "score": float(main_score),
            "subject": r.get("subject"),
            "chunk_file": r.get("chunk_file"),
            "id": r.get("id"),
        })

        header = (
            f"[{i}] subject={r.get('subject')} | "
            f"chunk_file={r.get('chunk_file')} | "
            f"id={r.get('id')} | score={main_score:.4f}\n"
        )
        blocks.append(header + r["text"])

    context = "\n\n---\n\n".join(blocks)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...[context truncated]"

    return context, citations


def get_system_prompt(lang: str) -> str:
    """Get system prompt based on language"""
    if lang == "zh":
        return (
            "你是计算机多课程智能问答助手，仅基于重排后的Top5知识库片段回答，需将教材生硬知识点润色为通顺语言，严格遵守以下规则：\n"
            "1. 知识来源：仅使用提供的上下文，不编造内容；\n"
            "2. 润色要求：句子简化、逻辑重组、补充必要细节，保留专业术语；\n"
            "3. 引用规则：先输出完整答案，末尾标注引用编号（如[1][3]），仅引用实际使用的片段；\n"
            "4. 异常处理：上下文不足时，仅输出'资料不足，无法回答该问题'；\n"
            "5. 格式要求：中文回答，简洁适配终端CLI展示，无冗余。"
        )
    else:
        return (
            "You are a multi-course computer science Q&A assistant. Answer ONLY using the reranked Top5 knowledge base snippets.\n"
            "Rules:\n"
            "1. Use ONLY the provided context, do not fabricate content.\n"
            "2. Simplify sentences, reorganize logic, add necessary details while keeping technical terms.\n"
            "3. Cite sources using [1][2] format at the end.\n"
            "4. If context is insufficient, output 'Insufficient information in provided materials.'\n"
            "5. Answer in English, concise for CLI display."
        )


class RAGService:
    """RAG Core Service - wraps retrieval and LLM generation logic"""

    def __init__(self):
        self.retriever = FaissRetriever()
        self.reranker = BGECrossEncoderReranker()

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        self.bm25 = BM25Retriever(self.docs)

        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)

    def _rag_answer(self, question: str, context: str, answer_lang: str) -> str:
        """Generate answer using LLM"""
        system_prompt = get_system_prompt(answer_lang)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Now produce the final answer."
        )

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    async def _rag_answer_stream(self, question: str, context: str, answer_lang: str) -> Generator[str, None, None]:
        """Stream answer using async LLM"""
        system_prompt = get_system_prompt(answer_lang)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Now produce the final answer."
        )

        stream = await self.async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _llm_retrieval(self, question: str, subject_filter: Optional[str] = None) -> tuple[List[Dict], str]:
        """LLM-based retrieval mode"""
        search_type = "dense+bm25"

        dense_results = self.retriever.search(question, top_k=RECALL_N)
        bm25_results = self.bm25.search(question, top_k=RECALL_N)

        docs_text = ""
        for i, r in enumerate(dense_results[:10], 1):
            docs_text += f"{i}. ID-{r['id']} 课程-{r.get('subject', 'N/A')} 文本-{r['text'][:200]}...\n"

        for i, r in enumerate(bm25_results[:10], 1):
            docs_text += f"{i+10}. ID-{r['id']} 课程-{r.get('subject', 'N/A')} 文本-{r['text'][:200]}...\n"

        system_prompt = (
            "你是面向计算机专业多课程（数据结构/操作系统/计算机网络）的检索模型，仅基于指定教材知识库片段检索，不引入外部知识，结果需严格匹配查询所属课程。\n"
            f"请按以下规则判定相关性并输出结构化结果，检索类型：【{search_type}】（Dense/dense+bm25）：\n"
            "1. 相关性规则（与rel3/2/1/0标注一致）：\n"
            "   - rel=3：含完整解答核心知识点；rel=2：含关键知识点但需补充；rel=1：主题相关无直接解答；rel=0：无关联。\n"
            "2. 检索信号：\n"
            "   - Dense检索：仅基于语义相似度；dense+bm25检索：均等融合BM25关键词匹配与Dense语义相似度。\n"
            "3. 相似度评分规则（0.00~1.00连续分数）：\n"
            "   - 0.90~1.00：文本完整解答查询，关键信息全覆盖（对应rel=3）\n"
            "   - 0.70~0.89：文本包含大部分关键信息，但需少量补充（对应rel=2）\n"
            "   - 0.40~0.69：文本主题相关，但缺少直接解答内容（对应rel=1）\n"
            "   - 0.00~0.39：文本与查询无实质关联（对应rel=0）\n"
            "4. 输出格式（严格JSON数组）：\n"
            '   [{"ID":"文本ID","课程":"DS/OS/CN","相关性":3,"匹配点":"关键词/语义描述","相似度":0.85}, ...]\n'
            f"5. 约束：若指定课程过滤【{subject_filter}】，仅检索该课程片段。"
        )

        user_prompt = (
            f"当前查询：{question}\n"
            f"知识库片段集：{docs_text}\n\n"
            "请按上述规则输出检索结果。"
        )

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        retrieval_result = resp.choices[0].message.content

        # Parse JSON from response
        json_match = re.search(r'\[.*\]', retrieval_result, re.DOTALL)
        if json_match:
            try:
                retrieval_data = json.loads(json_match.group())

                if len(retrieval_data) >= 5:
                    top_docs = retrieval_data[:5]
                    results = []
                    for doc_info in top_docs:
                        doc_id = doc_info.get("ID", "")
                        for r in dense_results:
                            if r["id"] == doc_id:
                                # Back-fill LLM scores into result dict
                                r = dict(r)  # shallow copy to avoid mutating original
                                r["llm_relevance"] = doc_info.get("相关性", 0)
                                llm_sim = doc_info.get("相似度", None)
                                r["llm_score"] = float(llm_sim) if llm_sim is not None else None
                                r["llm_match"] = doc_info.get("匹配点", "")
                                results.append(r)
                                break

                    if results:
                        return results, retrieval_result
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return dense_results[:TOP_K], retrieval_result

    def _search_dense_custom(self, question: str, top_k: int, use_bm25: bool, use_rerank: bool) -> List[Dict]:
        """Dense retrieval with configurable BM25 fusion and reranking"""
        if use_bm25:
            # Dense + BM25 hybrid with RRF fusion
            dense_results = self.retriever.search(question, top_k=FUSION_K)
            bm25_results = self.bm25.search(question, top_k=FUSION_K)

            merged = {}
            for rank, r in enumerate(dense_results, 1):
                rid = r["id"]
                merged.setdefault(rid, dict(r))
                merged[rid]["dense_score"] = r.get("score")
                merged[rid]["dense_rank"] = rank
                merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

            for rank, r in enumerate(bm25_results, 1):
                rid = r["id"]
                merged.setdefault(rid, dict(r))
                merged[rid]["bm25_score"] = r.get("bm25_score")
                merged[rid]["bm25_rank"] = rank
                merged[rid]["rrf_score"] = merged[rid].get("rrf_score", 0.0) + rrf(rank)

            # Sort by RRF score, take enough candidates for potential reranking
            results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

            if use_rerank:
                # Rerank top candidates, then trim to top_k
                rerank_candidates = results[:RECALL_N]
                results = self.reranker.rerank(question, rerank_candidates, top_k=top_k)
            else:
                results = results[:top_k]
                for r in results:
                    r["score"] = r.get("dense_score") if r.get("dense_score") is not None else r.get("bm25_score", 0.0)
        else:
            # Dense only
            if use_rerank:
                candidates = self.retriever.search(question, top_k=RECALL_N)
                # Preserve dense score as dense_score for dual display
                for c in candidates:
                    c["dense_score"] = c.get("score")
                results = self.reranker.rerank(question, candidates, top_k=top_k)
            else:
                results = self.retriever.search(question, top_k=top_k)

        return results

    def query(self, question: str, mode: str = "llm_retrieval",
              top_k: int = 5, use_bm25: bool = False, use_rerank: bool = False) -> Dict:
        """
        Query RAG system

        Args:
            question: User question
            mode: Retrieval mode - "dense" or "llm_retrieval"
            top_k: Number of results to return (1-10)
            use_bm25: Enable BM25 hybrid retrieval (dense+bm25)
            use_rerank: Enable BGE reranking (requires use_bm25=True for meaningful use)

        Returns:
            Dict with keys: answer, retrieval, mode, lang, citations
        """
        if not question or not question.strip():
            return {"error": "问题不能为空"}

        # Clamp top_k to valid range
        top_k = max(1, min(10, top_k))

        answer_lang = detect_lang(question)
        results = []

        # Retrieve based on mode
        if mode == "llm_retrieval":
            results, _ = self._llm_retrieval(question)
        elif mode == "dense":
            results = self._search_dense_custom(question, top_k, use_bm25, use_rerank)
        else:
            return {"error": f"未知模式: {mode}"}

        if not results:
            return {"error": "未检索到相关文档", "retrieval": [], "mode": mode, "lang": answer_lang}

        # Build context and generate answer
        context, citations = build_context(results)

        try:
            answer = self._rag_answer(question, context, answer_lang)
        except Exception as e:
            return {"error": f"生成失败: {str(e)}", "retrieval": results, "mode": mode, "lang": answer_lang}

        return {
            "answer": answer,
            "retrieval": results,
            "citations": citations,
            "mode": mode,
            "lang": answer_lang,
        }

    async def query_stream(self, question: str, mode: str = "llm_retrieval",
                           top_k: int = 5, use_bm25: bool = False, use_rerank: bool = False) -> Generator[Dict, None, None]:
        """
        Stream RAG query response

        Args:
            question: User question
            mode: Retrieval mode - "dense" or "llm_retrieval"
            top_k: Number of results to return (1-10)
            use_bm25: Enable BM25 hybrid retrieval
            use_rerank: Enable BGE reranking

        Yields chunks with type: 'retrieval' or 'chunk' or 'done'
        """
        if not question or not question.strip():
            yield {"type": "error", "message": "问题不能为空"}
            return

        # Clamp top_k to valid range
        top_k = max(1, min(10, top_k))

        answer_lang = detect_lang(question)
        results = []

        # Retrieve based on mode
        if mode == "llm_retrieval":
            results, _ = self._llm_retrieval(question)
        elif mode == "dense":
            results = self._search_dense_custom(question, top_k, use_bm25, use_rerank)
        else:
            yield {"type": "error", "message": f"未知模式: {mode}"}
            return

        if not results:
            yield {"type": "error", "message": "未检索到相关文档"}
            return

        # Send retrieval results first
        context, citations = build_context(results)
        yield {
            "type": "retrieval",
            "retrieval": results,
            "citations": citations,
            "mode": mode,
            "lang": answer_lang,
        }

        # Stream answer
        try:
            full_answer = ""
            async for chunk in self._rag_answer_stream(question, context, answer_lang):
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}

            yield {"type": "done", "answer": full_answer}
        except Exception as e:
            yield {"type": "error", "message": f"生成失败: {str(e)}"}


# Global service instance (singleton)
_service_instance = None


def get_rag_service() -> RAGService:
    """Get or create global RAGService instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = RAGService()
    return _service_instance
