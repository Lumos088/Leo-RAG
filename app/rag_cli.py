import sys
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scripts.pipeline.retriever import FaissRetriever
from scripts.pipeline.hybrid_retrieval import hybrid_search, rrf
from app.reranker import BGECrossEncoderReranker
from app.bm25_retriever import BM25Retriever

from openai import OpenAI

MODE = "dense"      # "dense", "dense_rerank", "dense_bm25"
RECALL_N = 50            # rerank 前的召回数量
TOP_K = 5
MAX_CONTEXT_CHARS = 8000
META_PATH = os.path.join(parent_dir, "vector_db", "kb_meta.json")

MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def detect_lang(text: str) -> str:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"

def get_main_score(r):
    return (
        r.get("rerank_score")
        if r.get("rerank_score") is not None else
        r.get("score")
        if r.get("score") is not None else
        r.get("bm25_score")
        if r.get("bm25_score") is not None else
        r.get("rrf_score", 0.0)
    )

def build_context(results):
    blocks = []
    citations = []

    for i, r in enumerate(results, 1):
        main_score = get_main_score(r)
        citations.append({
            "rank": i,
            "score": main_score,
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

def rag_answer(question: str, context: str, answer_lang: str):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )

    if answer_lang == "zh":
        system_prompt = (
            "你是计算机多课程智能问答助手，仅基于重排后的Top5知识库片段回答，需将教材生硬知识点润色为通顺语言，严格遵守以下规则：\n"
            "1. 知识来源：仅使用提供的上下文，不编造内容；\n"
            "2. 润色要求：句子简化、逻辑重组、补充必要细节，保留专业术语；\n"
            "3. 引用规则：先输出完整答案，末尾标注引用编号（如[1][3]），仅引用实际使用的片段；\n"
            "4. 异常处理：上下文不足时，仅输出'资料不足，无法回答该问题'；\n"
            "5. 格式要求：中文回答，简洁适配终端CLI展示，无冗余。"
        )
    else:
        system_prompt = (
            "You are a multi-course computer science Q&A assistant. Answer ONLY using the reranked Top5 knowledge base snippets.\n"
            "Rules:\n"
            "1. Use ONLY the provided context, do not fabricate content.\n"
            "2. Simplify sentences, reorganize logic, add necessary details while keeping technical terms.\n"
            "3. Cite sources using [1][2] format at the end.\n"
            "4. If context is insufficient, output 'Insufficient information in provided materials.'\n"
            "5. Answer in English, concise for CLI display."
        )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Now produce the final answer."
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content

def main():
    retriever = FaissRetriever()
    reranker = BGECrossEncoderReranker() if MODE == "dense_rerank" else None

    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    bm25_retriever = BM25Retriever(docs) if MODE == "dense_bm25" else None

    print(f"\n✅ RAG CLI（DeepSeek）启动成功 | 模式: {MODE}")
    print("输入问题，exit 退出。\n")

    while True:
        question = input(">> ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        answer_lang = detect_lang(question)

        if MODE == "dense":
            results = retriever.search(question, top_k=TOP_K)
        elif MODE == "dense_rerank":
            candidates = retriever.search(question, top_k=RECALL_N)
            results = reranker.rerank(question, candidates, top_k=TOP_K)
        elif MODE == "dense_bm25":
            results = hybrid_search(retriever, bm25_retriever, question)[:TOP_K]

            for r in results:
                r["score"] = r.get("dense_score") if r.get("dense_score") is not None else r.get("bm25_score", 0.0)
        else:
            raise ValueError("MODE must be 'dense', 'dense_rerank' or 'dense_bm25'")

        context, citations = build_context(results)

        try:
            answer = rag_answer(question, context, answer_lang)
        except Exception as e:
            print("\n❌ 生成失败：", e)
            continue

        print("\n================= Top-k Retrieval =================")
        for i, r in enumerate(results, 1):
            msg = f"[{i}] dense_score={r.get('dense_score', r.get('score', 0.0)):.4f}"
            if r.get("bm25_score") is not None:
                msg += f" bm25_score={r['bm25_score']:.4f}"
            if r.get("rerank_score") is not None:
                msg += f" rerank_score={r['rerank_score']:.4f}"
            if r.get("rrf_score") is not None:
                msg += f" rrf={r['rrf_score']:.6f}"
            msg += f" subject={r.get('subject')} chunk_file={r.get('chunk_file')} id={r.get('id')}"
            print(msg)

            if MODE == "dense_bm25":
                dense_rank = r.get("dense_rank")
                bm25_rank = r.get("bm25_rank")
                dense_rank_str = str(dense_rank) if dense_rank is not None else "∞"
                bm25_rank_str = str(bm25_rank) if bm25_rank is not None else "∞"

                rrf_dense = rrf(dense_rank) if dense_rank is not None else 0.0
                rrf_bm25 = rrf(bm25_rank) if bm25_rank is not None else 0.0

                print(f"    dense 排名: {dense_rank_str}, bm25 排名: {bm25_rank_str}")
                print(f"    rrf_score = rrf({dense_rank_str}) + rrf({bm25_rank_str}) = {rrf_dense:.4f} + {rrf_bm25:.4f} = {r['rrf_score']:.6f}")
        print("\n========== RAG Answer ==========")
        print(answer)
        print("\n")

if __name__ == "__main__":
    main()
