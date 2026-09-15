"""
RAG Core Service
Extracted from app/rag_cli.py for Web UI usage
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Generator
from openai import OpenAI, AsyncOpenAI

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
stage2_dir = os.path.dirname(parent_dir)
sys.path.append(stage2_dir)

from scripts.pipeline.retriever import FaissRetriever
from scripts.pipeline.hybrid_retrieval import (
    RETRIEVAL_CONFIG,
    fuse_results,
    hybrid_rerank,
    hybrid_search,
)
from app.reranker import BGECrossEncoderReranker
from app.bm25_retriever import BM25Retriever
from scripts.pipeline.parent_child_retrieval import ParentChildRetriever, load_parent_child_config
from scripts.pipeline.query_router import detect_subject
from scripts.pipeline.context_assembly import (
    build_parent_context,
    infer_query_type,
    insert_baseline_rescue,
)
from web_ui.api.memory import ConversationMemory

# Stage-1 production configuration. The original vector_db remains untouched and
# can be restored by changing index_dir in config/retrieval.json.
PROJECT_DIR = Path(stage2_dir)
DEFAULT_TOP_K = int(RETRIEVAL_CONFIG["final_top_k"])
MAX_TOP_K = int(RETRIEVAL_CONFIG["max_top_k"])
MAX_CONTEXT_CHARS = int(RETRIEVAL_CONFIG["max_context_chars"])

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
            "course": r.get("course"),
            "document_title": r.get("document_title"),
            "chapter_path": r.get("chapter_path"),
            "page": r.get("page") or r.get("page_number"),
            "chunk_file": r.get("chunk_file"),
            "id": r.get("id"),
        })

        source_parts = [
            f"subject={r.get('subject')}",
            f"course={r.get('course')}",
            f"document={r.get('document_title')}",
            f"chapter={r.get('chapter_path')}",
            f"page={r.get('page') or r.get('page_number')}",
            f"chunk_file={r.get('chunk_file')}",
            f"id={r.get('id')}",
            f"score={main_score:.4f}",
        ]
        header = (
            f"[{i}] " + " | ".join(part for part in source_parts if not part.endswith("=None")) + "\n"
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
            "你是计算机课程知识库问答助手。请严格遵守：\n"
            "1. 只能依据给定资料回答，不得补充资料之外的事实；资料不足时明确说明。\n"
            "2. 直接回答问题，结构清晰、简洁，保留必要的专业术语。\n"
            "3. 每个关键结论后标注支持它的资料编号，如 [1] 或 [1][3]。\n"
            "4. 不要引用没有实际支持对应结论的资料。使用中文回答。"
        )
    else:
        return (
            "You are a multi-course computer science Q&A assistant. Answer ONLY using the provided knowledge base snippets.\n"
            "Rules:\n"
            "1. Use ONLY the provided context, do not fabricate content.\n"
            "2. Simplify sentences, reorganize logic, add necessary details while keeping technical terms.\n"
            "3. Cite each key claim using the supporting source number, such as [1][2].\n"
            "4. If context is insufficient, output 'Insufficient information in provided materials.'\n"
            "5. Answer in English, concise for CLI display."
        )


class RAGService:
    """RAG Core Service - wraps retrieval and LLM generation logic"""

    def __init__(self):
        index_dir = Path(RETRIEVAL_CONFIG["index_dir"])
        if not index_dir.is_absolute():
            index_dir = PROJECT_DIR / index_dir
        self.index_dir = index_dir.resolve()
        self.retriever = FaissRetriever(index_dir=self.index_dir)
        self.reranker = BGECrossEncoderReranker()
        # BM25 and FAISS must use the same versioned corpus and metadata.
        self.docs = self.retriever.metadatas
        self.bm25 = BM25Retriever(self.docs)
        self.parent_child_config = load_parent_child_config(
            PROJECT_DIR / "config" / "parent_child.json"
        )
        self.parent_child = None
        if self.parent_child_config.get("enabled"):
            self.parent_child = ParentChildRetriever(
                config_path=PROJECT_DIR / "config" / "parent_child.json",
                reranker=self.reranker,
            )

        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
        self.memory = ConversationMemory(self.client, MODEL_NAME)

    def _rag_answer(self, question: str, context: str, answer_lang: str,
                    conversation_context: str = "") -> str:
        """Generate answer using LLM"""
        system_prompt = get_system_prompt(answer_lang)
        memory_block = (
            f"Conversation memory (only resolve references; do not treat it as evidence):\n"
            f"{conversation_context}\n\n" if conversation_context else ""
        )
        user_prompt = (
            memory_block +
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

    async def _rag_answer_stream(self, question: str, context: str, answer_lang: str,
                                 conversation_context: str = "") -> Generator[str, None, None]:
        """Stream answer using async LLM"""
        system_prompt = get_system_prompt(answer_lang)
        memory_block = (
            f"Conversation memory (only resolve references; do not treat it as evidence):\n"
            f"{conversation_context}\n\n" if conversation_context else ""
        )
        user_prompt = (
            memory_block +
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

    def _search_dense_custom(self, question: str, top_k: int, use_bm25: bool, use_rerank: bool) -> List[Dict]:
        """Dense retrieval with configurable BM25 fusion and reranking"""
        if use_bm25:
            if use_rerank:
                results = hybrid_rerank(
                    self.retriever, self.bm25, self.reranker, question, top_k
                )
            else:
                results = hybrid_search(self.retriever, self.bm25, question)[:top_k]
                for r in results:
                    r["score"] = r.get("dense_score") if r.get("dense_score") is not None else r.get("bm25_score", 0.0)
        else:
            # Dense only
            if use_rerank:
                candidates = self.retriever.search(
                    question, top_k=RETRIEVAL_CONFIG["rerank_candidates"]
                )
                # Preserve dense score as dense_score for dual display
                for c in candidates:
                    c["dense_score"] = c.get("score")
                results = self.reranker.rerank(question, candidates, top_k=top_k)
            else:
                results = self.retriever.search(question, top_k=top_k)

        return results

    def _search_parent_child_custom(
        self, question: str, top_k: int, use_bm25: bool, use_rerank: bool
    ) -> tuple[List[Dict], str, List[Dict]]:
        """Run the validated adaptive Parent-Child context policy."""
        if self.parent_child is None:
            raise RuntimeError("Parent-Child retrieval is not enabled")
        parent_result = self.parent_child.search(
            question,
            parent_top_k=top_k,
            use_bm25=use_bm25,
            use_rerank=use_rerank,
        )
        parents = list(parent_result["parents"])
        baseline = self._search_dense_custom(question, top_k, use_bm25, use_rerank)
        if (
            self.parent_child_config.get("baseline_rescue_enabled")
            and detect_subject(question) != "Artificial Intelligence"
        ):
            parents, _ = insert_baseline_rescue(
                parents, baseline, self.parent_child.parents
            )
        highlights = {}
        query_type = infer_query_type(question)
        if (
            query_type == "definition"
            and use_rerank
            and self.parent_child_config.get("definition_child_highlight_enabled")
        ):
            highlights = self.parent_child.select_parent_highlights(
                question,
                [str(parent["id"]) for parent in parents],
                per_parent=self.parent_child_config["child_highlights_per_parent"],
                use_rerank=True,
            )
        context, citations, retrieval = build_parent_context(
            parents,
            max_chars=self.parent_child_config["max_context_chars"],
            highlights=highlights,
        )
        for item in retrieval:
            item["query_type"] = query_type
            item["retrieval_strategy"] = "parent_child_adaptive"
        return retrieval, context, citations

    def query(self, question: str, mode: str = "dense",
              top_k: int = DEFAULT_TOP_K, use_bm25: bool = True, use_rerank: bool = True,
              retrieval_question: str | None = None,
              conversation_context: str = "") -> Dict:
        """
        Query RAG system

        Args:
            question: User question
            mode: Retrieval mode; only "dense" is supported
            top_k: Number of results to return (1-MAX_TOP_K)
            use_bm25: Enable BM25 hybrid retrieval (dense+bm25)
            use_rerank: Enable BGE reranking (requires use_bm25=True for meaningful use)

        Returns:
            Dict with keys: answer, retrieval, mode, lang, citations
        """
        if not question or not question.strip():
            return {"error": "问题不能为空"}

        # Clamp top_k to valid range
        top_k = max(1, min(MAX_TOP_K, top_k))

        answer_lang = detect_lang(question)
        search_question = retrieval_question or question
        results = []

        # Retrieve based on mode
        context = None
        citations = None
        if mode == "dense" and self.parent_child is not None:
            results, context, citations = self._search_parent_child_custom(
                search_question, top_k, use_bm25, use_rerank
            )
        elif mode == "dense":
            results = self._search_dense_custom(search_question, top_k, use_bm25, use_rerank)
        else:
            return {"error": f"未知模式: {mode}"}

        if not results:
            return {"error": "未检索到相关文档", "retrieval": [], "mode": mode, "lang": answer_lang}

        # Build context and generate answer
        if context is None or citations is None:
            context, citations = build_context(results)

        try:
            answer = self._rag_answer(question, context, answer_lang, conversation_context)
        except Exception as e:
            return {"error": f"生成失败: {str(e)}", "retrieval": results, "mode": mode, "lang": answer_lang}

        return {
            "answer": answer,
            "retrieval": results,
            "citations": citations,
            "mode": mode,
            "lang": answer_lang,
            "retrieval_query": search_question,
        }

    async def _query_stream_legacy(self, question: str, mode: str = "dense",
                           top_k: int = DEFAULT_TOP_K, use_bm25: bool = True, use_rerank: bool = True) -> Generator[Dict, None, None]:
        """
        Stream RAG query response

        Args:
            question: User question
            mode: Retrieval mode; only "dense" is supported
            top_k: Number of results to return (1-10)
            use_bm25: Enable BM25 hybrid retrieval
            use_rerank: Enable BGE reranking

        Yields chunks with type: 'retrieval' or 'chunk' or 'done'
        """
        if not question or not question.strip():
            yield {"type": "error", "message": "问题不能为空"}
            return

        # Clamp top_k to valid range
        top_k = max(1, min(MAX_TOP_K, top_k))

        answer_lang = detect_lang(question)
        results = []

        # Retrieve based on mode
        if mode == "dense":
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

    async def query_stream(self, question: str, mode: str = "dense",
                           top_k: int = DEFAULT_TOP_K, use_bm25: bool = True,
                           use_rerank: bool = True,
                           retrieval_question: str | None = None,
                           conversation_context: str = "") -> Generator[Dict, None, None]:
        """Stream backend-synchronized pipeline progress and answer events."""
        if not question or not question.strip():
            yield {"type": "error", "message": "Question cannot be empty"}
            return

        top_k = max(1, min(MAX_TOP_K, top_k))
        answer_lang = detect_lang(question)
        search_question = retrieval_question or question
        yield {"type": "progress", "stage": "query"}

        if mode != "dense":
            yield {"type": "error", "message": f"Unknown mode: {mode}"}
            return

        context = None
        citations = None
        if self.parent_child is not None:
            yield {"type": "progress", "stage": "dense_retrieval"}
            if use_bm25:
                yield {"type": "progress", "stage": "bm25_retrieval"}
                yield {"type": "progress", "stage": "fusion"}
            if use_rerank:
                yield {"type": "progress", "stage": "bge_rerank"}
            results, context, citations = self._search_parent_child_custom(
                search_question, top_k, use_bm25, use_rerank
            )
        elif use_bm25:
            yield {"type": "progress", "stage": "dense_retrieval"}
            dense_results = self.retriever.search(
                search_question, top_k=RETRIEVAL_CONFIG["dense_k"]
            )

            yield {"type": "progress", "stage": "bm25_retrieval"}
            bm25_results = self.bm25.search(
                search_question, top_k=RETRIEVAL_CONFIG["bm25_k"]
            )

            yield {"type": "progress", "stage": "fusion"}
            results = fuse_results(dense_results, bm25_results)
            if use_rerank:
                yield {"type": "progress", "stage": "bge_rerank"}
                candidates = results[:RETRIEVAL_CONFIG["rerank_candidates"]]
                results = self.reranker.rerank(search_question, candidates, top_k=top_k)
            else:
                results = results[:top_k]
                for result in results:
                    result["score"] = (
                        result.get("dense_score")
                        if result.get("dense_score") is not None
                        else result.get("bm25_score", 0.0)
                    )
        else:
            yield {"type": "progress", "stage": "dense_retrieval"}
            search_k = (
                RETRIEVAL_CONFIG["rerank_candidates"] if use_rerank else top_k
            )
            results = self.retriever.search(search_question, top_k=search_k)
            if use_rerank:
                for result in results:
                    result["dense_score"] = result.get("score")
                yield {"type": "progress", "stage": "bge_rerank"}
                results = self.reranker.rerank(search_question, results, top_k=top_k)

        if not results:
            yield {"type": "error", "message": "No relevant documents found"}
            return

        yield {"type": "progress", "stage": "context_assembly"}
        if context is None or citations is None:
            context, citations = build_context(results)
        yield {
            "type": "retrieval",
            "retrieval": results,
            "citations": citations,
            "mode": mode,
            "lang": answer_lang,
            "retrieval_query": search_question,
        }

        try:
            yield {"type": "progress", "stage": "llm_generation"}
            full_answer = ""
            async for chunk in self._rag_answer_stream(
                question, context, answer_lang, conversation_context
            ):
                full_answer += chunk
                yield {"type": "chunk", "content": chunk}

            yield {"type": "progress", "stage": "answer"}
            yield {"type": "done", "answer": full_answer}
        except Exception as exc:
            yield {"type": "error", "message": f"Generation failed: {exc}"}


# Global service instance (singleton)
_service_instance = None


def get_rag_service() -> RAGService:
    """Get or create global RAGService instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = RAGService()
    return _service_instance
