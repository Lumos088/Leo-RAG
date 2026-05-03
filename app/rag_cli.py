import sys
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scripts.pipeline.retriever import FaissRetriever
from app.reranker import BGECrossEncoderReranker
from app.bm25_retriever import BM25Retriever

from openai import OpenAI

MODE = "dense"      # "dense", "dense_rerank", "dense_bm25", "llm_retrieval"
RECALL_N = 50            # rerank 前的召回数量
FUSION_K = 20            # dense + bm25 融合时的召回数量
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

def rrf(rank, k=60):
    return 1.0 / (k + rank)

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

def llm_retrieval(question: str, retriever, bm25_retriever, subject_filter=None):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )

    search_type = "dense+bm25" if bm25_retriever else "Dense"

    dense_results = retriever.search(question, top_k=RECALL_N)
    if bm25_retriever:
        bm25_results = bm25_retriever.search(question, top_k=RECALL_N)

    docs_text = ""
    for i, r in enumerate(dense_results[:10], 1):
        docs_text += f"{i}. ID-{r['id']} 课程-{r.get('subject', 'N/A')} 文本-{r['text'][:200]}...\n"

    if bm25_retriever:
        for i, r in enumerate(bm25_results[:10], 1):
            docs_text += f"{i+10}. ID-{r['id']} 课程-{r.get('subject', 'N/A')} 文本-{r['text'][:200]}...\n"

    system_prompt = (
        "你是面向计算机专业多课程（数据结构/操作系统/计算机网络）的检索模型，仅基于指定教材知识库片段检索，不引入外部知识，结果需严格匹配查询所属课程。\n"
        f"请按以下规则判定相关性并输出结构化结果，检索类型：【{search_type}】（Dense/dense+bm25）：\n"
        "1. 相关性规则（与rel3/2/1/0标注一致）：\n"
        "   - rel=3：含完整解答核心知识点；rel=2：含关键知识点但需补充；rel=1：主题相关无直接解答；rel=0：无关联。\n"
        "2. 检索信号：\n"
        "   - Dense检索：仅基于语义相似度；dense+bm25检索：均等融合BM25关键词匹配与Dense语义相似度。\n"
        "3. 输出格式：ID-[文本ID] 课程-[DS/OS/CN] 相关性-[3/2/1/0] 匹配点-[关键词/语义描述] 相似度-[分数]\n"
        f"4. 约束：若指定课程过滤【{subject_filter}】，仅检索该课程片段。"
    )

    user_prompt = (
        f"当前查询：{question}\n"
        f"知识库片段集：{docs_text}\n\n"
        "请按上述规则输出检索结果。"
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return resp.choices[0].message.content, dense_results

def main():
    retriever = FaissRetriever()
    reranker = BGECrossEncoderReranker() if MODE == "dense_rerank" else None

    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    bm25_retriever = BM25Retriever(docs) if MODE in ["dense_bm25", "llm_retrieval"] else None

    print(f"\n✅ RAG CLI（DeepSeek）启动成功 | 模式: {MODE}")
    print("输入问题，exit 退出。\n")

    while True:
        question = input(">> ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        answer_lang = detect_lang(question)

        if MODE == "llm_retrieval":
            print("\n========== LLM 检索 ==========")
            
            retrieval_result, dense_results = llm_retrieval(question, retriever, bm25_retriever, subject_filter=None)
            print(retrieval_result)
            
            print("\n========== 最终结果 ==========")
            print("基于 LLM 检索结果生成回答...")
            
            try:
                import re
                
                json_match = re.search(r'\[.*\]', retrieval_result, re.DOTALL)
                if json_match:
                    retrieval_data = json.loads(json_match.group())
                    
                    if len(retrieval_data) >= 5:
                        top_docs = retrieval_data[:5]
                        
                        results = []
                        for doc_info in top_docs:
                            doc_id = doc_info.get("ID", "")
                            for r in dense_results:
                                if r["id"] == doc_id:
                                    results.append(r)
                                    break
                        
                        if len(results) > 0:
                            context, citations = build_context(results)
                            answer = rag_answer(question, context, answer_lang)
                            
                            print("\n================= Top-k Retrieval =================")
                            for i, r in enumerate(results, 1):
                                msg = f"[{i}] score={r.get('score', 0.0):.4f}"
                                msg += f" subject={r.get('subject')} chunk_file={r.get('chunk_file')} id={r.get('id')}"
                                print(msg)
                            
                            print("\n================= Answer =================")
                            print(answer)
                        else:
                            print("❌ 未能从检索结果中找到匹配的文档")
                    else:
                        print(f"❌ 检索结果不足5个，仅找到 {len(retrieval_data)} 个")
                else:
                    print("❌ 未能解析检索结果中的JSON数据")
            except Exception as e:
                print(f"❌ 解析检索结果失败：{e}")
            
            continue

        if MODE == "dense":
            results = retriever.search(question, top_k=TOP_K)
        elif MODE == "dense_rerank":
            candidates = retriever.search(question, top_k=RECALL_N)
            results = reranker.rerank(question, candidates, top_k=TOP_K)
        elif MODE == "dense_bm25":
            dense_results = retriever.search(question, top_k=FUSION_K)
            bm25_results = bm25_retriever.search(question, top_k=FUSION_K)

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

            results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:TOP_K]

            for r in results:
                r["score"] = r.get("dense_score") if r.get("dense_score") is not None else r.get("bm25_score", 0.0)
        else:
            raise ValueError("MODE must be 'dense', 'dense_rerank', 'dense_bm25' or 'llm_retrieval'")

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
