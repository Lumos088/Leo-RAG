import sys
import os
import csv
import json
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from scripts.pipeline.retriever import FaissRetriever
from scripts.pipeline.scibert_retriever import SciBertRetriever
from app.bm25_retriever import BM25Retriever
from app.reranker import BGECrossEncoderReranker

from openai import OpenAI

TOPICS_PATH = os.path.join(parent_dir, "topics.csv")
META_PATH = os.path.join(parent_dir, "vector_db", "kb_meta.json")
RUN_PATH = os.path.join(parent_dir, "runs")

FUSION_K = 50
TOP_K = 50
RECALL_N = 50

MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 读取文档内容
def load_doc_contents():
    doc_contents = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
        for doc in docs:
            doc_id = doc["id"]
            doc_contents[doc_id] = doc.get("text", "")
    return doc_contents

def rrf(rank, k=60):
    return 1.0 / (k + rank)

def load_topics():
    topics = []
    with open(TOPICS_PATH, "r", encoding="gbk") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append({
                "qid": row["qid"],
                "query": row["query"],
                "type": row["type"]
            })
    return topics

def run_dense(retriever, query, top_k=TOP_K):
    results = retriever.search(query, top_k=top_k)
    return results

def run_dense_bm25(retriever, bm25_retriever, query, top_k=TOP_K, fusion_k=FUSION_K):
    dense_results = retriever.search(query, top_k=fusion_k)
    bm25_results = bm25_retriever.search(query, top_k=fusion_k)

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

    results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
    return results

def run_rerank(retriever, bm25_retriever, reranker, doc_contents, query, top_k=TOP_K, fusion_k=FUSION_K):
    # 先获取融合结果
    dense_results = retriever.search(query, top_k=fusion_k)
    bm25_results = bm25_retriever.search(query, top_k=fusion_k)

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

    candidates = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:fusion_k]
    
    # 添加文档内容用于 rerank
    for c in candidates:
        c["text"] = doc_contents.get(c["id"], "")
    
    # rerank
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    return reranked

def run_llm_retrieval(retriever, bm25_retriever, query, top_k=TOP_K):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )

    search_type = "dense+bm25" if bm25_retriever else "Dense"

    dense_results = retriever.search(query, top_k=RECALL_N)
    if bm25_retriever:
        bm25_results = bm25_retriever.search(query, top_k=RECALL_N)

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
        "4. 约束：仅检索该课程片段。"
    )

    user_prompt = (
        f"当前查询：{query}\n"
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

    retrieval_result = resp.choices[0].message.content
    
    # 解析 LLM 返回的结果
    json_match = re.search(r'\[.*\]', retrieval_result, re.DOTALL)
    if json_match:
        try:
            retrieval_data = json.loads(json_match.group())
            
            results = []
            for doc_info in retrieval_data[:top_k]:
                doc_id = doc_info.get("ID", "")
                for r in dense_results:
                    if r["id"] == doc_id:
                        results.append(r)
                        break
            
            return results
        except json.JSONDecodeError as e:
            print(f"    [WARNING] LLM 返回的 JSON 解析失败: {e}")
            print(f"    [DEBUG] LLM 返回内容: {retrieval_result[:200]}...")
            return []
    else:
        print(f"    [WARNING] LLM 返回内容中未找到 JSON 格式")
        print(f"    [DEBUG] LLM 返回内容: {retrieval_result[:200]}...")
        return []

def save_run(results, qid, system_name, run_path):
    os.makedirs(run_path, exist_ok=True)
    run_file = os.path.join(run_path, f"{system_name}.run")
    
    mode = "a" if os.path.exists(run_file) else "w"
    
    with open(run_file, mode, encoding="utf-8") as f:
        for rank, r in enumerate(results, 1):
            if system_name in ["dense", "scibert"]:
                score = r.get("score", 0.0)
            elif system_name in ["dense_bm25", "scibert_bm25"]:
                score = r.get("rrf_score", 0.0)
            elif system_name in ["rerank", "scibert_rerank"]:
                score = r.get("rerank_score", 0.0)
            elif system_name in ["llm_retrieval"]:
                score = r.get("rrf_score", 0.0)
            else:
                score = 0.0
            
            doc_id = r["id"]
            f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {system_name}\n")

def main():
    print("=" * 60)
    print("批量检索脚本 - 生成 TREC run 文件")
    print("=" * 60)
    
    print("\n[1/5] 加载 topics...")
    topics = load_topics()
    print(f"    共加载 {len(topics)} 个问题")
    
    print("\n[2/5] 初始化检索器...")
    retriever = FaissRetriever()
    print("    Dense 检索器已初始化")
    
    try:
        scibert_retriever = SciBertRetriever()
        print("    SciBERT 检索器已初始化")
        scibert_available = True
    except FileNotFoundError as e:
        print(f"    SciBERT 检索器初始化失败: {e}")
        print("    将跳过 SciBERT 相关实验")
        scibert_available = False
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    bm25_retriever = BM25Retriever(docs)
    print("    BM25 检索器已初始化")
    
    reranker = BGECrossEncoderReranker()
    print("    Reranker 已初始化")
    
    print("\n[3/5] 加载文档内容...")
    doc_contents = load_doc_contents()
    print(f"    共加载 {len(doc_contents)} 个文档")
    
    print("\n[4/5] 批量检索并保存结果...")
    for i, topic in enumerate(topics, 1):
        qid = topic["qid"]
        query = topic["query"]
        
        print(f"    [{i}/{len(topics)}] {qid}: {query[:30]}..." if len(query) > 30 else f"    [{i}/{len(topics)}] {qid}: {query}")
        
        # Dense 检索
        dense_results = run_dense(retriever, query, top_k=TOP_K)
        save_run(dense_results, qid, "dense", RUN_PATH)
        
        # Dense + BM25 融合
        dense_bm25_results = run_dense_bm25(retriever, bm25_retriever, query, top_k=TOP_K, fusion_k=FUSION_K)
        save_run(dense_bm25_results, qid, "dense_bm25", RUN_PATH)
        
        # Rerank
        rerank_results = run_rerank(retriever, bm25_retriever, reranker, doc_contents, query, top_k=TOP_K, fusion_k=FUSION_K)
        save_run(rerank_results, qid, "rerank", RUN_PATH)
        
        # SciBERT 检索
        if scibert_available:
            scibert_results = run_dense(scibert_retriever, query, top_k=TOP_K)
            save_run(scibert_results, qid, "scibert", RUN_PATH)
            
            # SciBERT + BM25 融合
            scibert_bm25_results = run_dense_bm25(scibert_retriever, bm25_retriever, query, top_k=TOP_K, fusion_k=FUSION_K)
            save_run(scibert_bm25_results, qid, "scibert_bm25", RUN_PATH)
            
            # SciBERT + Rerank
            scibert_rerank_results = run_rerank(scibert_retriever, bm25_retriever, reranker, doc_contents, query, top_k=TOP_K, fusion_k=FUSION_K)
            save_run(scibert_rerank_results, qid, "scibert_rerank", RUN_PATH)
        
        # LLM Retrieval
        llm_retrieval_results = run_llm_retrieval(retriever, bm25_retriever, query, top_k=TOP_K)
        save_run(llm_retrieval_results, qid, "llm_retrieval", RUN_PATH)
        
    
    print("\n[5/5] 完成！")
    print(f"run 文件保存在: {RUN_PATH}")
    print("  - dense.run")
    print("  - dense_bm25.run")
    print("  - rerank.run")
    if scibert_available:
        print("  - scibert.run")
        print("  - scibert_bm25.run")
        print("  - scibert_rerank.run")
    print("  - llm_retrieval.run")
    print("=" * 60)

if __name__ == "__main__":
    main()
