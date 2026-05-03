import os
import csv
import math
from collections import defaultdict

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QRELS_PATH = os.path.join(parent_dir, "qrels.csv")
RUNS_DIR = os.path.join(parent_dir, "runs")

def load_qrels(path):
    qrels_binary = defaultdict(set)
    qrels_graded = defaultdict(dict)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk") as f:
            lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("qid,doc_id,rel"):
                start_idx = i + 1
                break
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            qid = parts[0]
            rel_str = parts[-2] if parts[-1] not in ["bm25", "dense", "dense_bm25", "tfidf"] else parts[-2]
            try:
                rel = int(float(rel_str))
            except:
                continue
            doc_id = ",".join(parts[1:-2]) if len(parts) > 3 else parts[1]
            doc_id = doc_id.strip('"')
            if rel > 0:
                qrels_binary[qid].add(doc_id)
            qrels_graded[qid][doc_id] = rel
    return qrels_binary, qrels_graded

def load_run(path):
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            doc_id = " ".join(parts[2:-3])
            rank = int(parts[-3])
            score = float(parts[-2])
            run[qid].append({"doc_id": doc_id, "rank": rank, "score": score})
    return run

def calc_mrr_at_k(run_results, qrels, k=10):
    mrr_sum = 0.0
    num_queries = 0
    
    for qid, results in run_results.items():
        if qid not in qrels:
            continue
        num_queries += 1
        relevant_docs = qrels[qid]
        
        for r in sorted(results, key=lambda x: x["rank"]):
            if r["rank"] > k:
                break
            if r["doc_id"] in relevant_docs:
                mrr_sum += 1.0 / r["rank"]
                break
    
    return mrr_sum / num_queries if num_queries > 0 else 0.0

def calc_success_at_k(run_results, qrels, k=5):
    success_count = 0
    num_queries = 0
    
    for qid, results in run_results.items():
        if qid not in qrels:
            continue
        num_queries += 1
        relevant_docs = qrels[qid]
        
        top_k_docs = set()
        for r in sorted(results, key=lambda x: x["rank"]):
            if r["rank"] <= k:
                top_k_docs.add(r["doc_id"])
        
        if top_k_docs & relevant_docs:
            success_count += 1
    
    return success_count / num_queries if num_queries > 0 else 0.0

def calc_recall_at_k(run_results, qrels, k=50):
    recall_sum = 0.0
    num_queries = 0
    
    for qid, results in run_results.items():
        if qid not in qrels:
            continue
        num_queries += 1
        relevant_docs = qrels[qid]
        
        top_k_docs = set()
        for r in sorted(results, key=lambda x: x["rank"]):
            if r["rank"] <= k:
                top_k_docs.add(r["doc_id"])
        
        hits = len(top_k_docs & relevant_docs)
        recall_sum += hits / len(relevant_docs) if relevant_docs else 0.0
    
    return recall_sum / num_queries if num_queries > 0 else 0.0

def dcg(scores):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))

def calc_ndcg_at_k(run_results, qrels_graded, k=10):
    ndcg_sum = 0.0
    num_queries = 0
    
    for qid, results in run_results.items():
        if qid not in qrels_graded:
            continue
        num_queries += 1
        rel_dict = qrels_graded[qid]
        
        gains = []
        for r in sorted(results, key=lambda x: x["rank"]):
            if r["rank"] > k:
                break
            rel = rel_dict.get(r["doc_id"], 0)
            gains.append(rel)
        
        ideal_gains = sorted(gains, reverse=True)
        
        dcg_val = dcg(gains)
        idcg_val = dcg(ideal_gains)
        
        if idcg_val > 0:
            ndcg_sum += dcg_val / idcg_val
    
    return ndcg_sum / num_queries if num_queries > 0 else 0.0

def main():
    print("=" * 80)
    print("IR 评估脚本 - dense vs dense_bm25 vs rerank vs scibert vs scibert_bm25 vs scibert_rerank vs llm_retrieval 评测")
    print("=" * 80)
    
    print("\n[1/3] 加载 qrels...")
    qrels_binary, qrels_graded = load_qrels(QRELS_PATH)
    print(f"    共加载 {len(qrels_binary)} 个查询的相关性标注")
    
    print("\n[2/3] 加载 run 文件...")
    runs = {}
    for system_name in ["dense", "dense_bm25", "rerank", "scibert", "scibert_bm25", "scibert_rerank", "llm_retrieval"]:
        run_path = os.path.join(RUNS_DIR, f"{system_name}.run")
        if os.path.exists(run_path):
            runs[system_name] = load_run(run_path)
            print(f"    {system_name}: {len(runs[system_name])} 个查询")
    
    print("\n[3/3] 计算评估指标...")
    print("\n" + "-" * 80)
    print(f"{'System':<15} {'MRR@10':<12} {'S@1':<12} {'S@5':<12} {'S@10':<12} {'R@50':<12} {'NDCG@10':<12} {'NDCG@50':<12}")
    print("-" * 80)
    
    results_table = {}
    for system_name, run_results in runs.items():
        mrr_10 = calc_mrr_at_k(run_results, qrels_binary, k=10)
        success_1 = calc_success_at_k(run_results, qrels_binary, k=1)
        success_5 = calc_success_at_k(run_results, qrels_binary, k=5)
        success_10 = calc_success_at_k(run_results, qrels_binary, k=10)
        recall_50 = calc_recall_at_k(run_results, qrels_binary, k=50)
        ndcg_10 = calc_ndcg_at_k(run_results, qrels_graded, k=10)
        ndcg_50 = calc_ndcg_at_k(run_results, qrels_graded, k=50)
        
        results_table[system_name] = {
            "MRR@10": mrr_10,
            "Success@1": success_1,
            "Success@5": success_5,
            "Success@10": success_10,
            "Recall@50": recall_50,
            "NDCG@10": ndcg_10,
            "NDCG@50": ndcg_50
        }
        
        print(f"{system_name:<15} {mrr_10:<12.4f} {success_1:<12.4f} {success_5:<12.4f} {success_10:<12.4f} "
              f"{recall_50:<12.4f} {ndcg_10:<12.4f} {ndcg_50:<12.4f}")
    
    print("-" * 80)
    
    if len(runs) >= 2:
        print("\n对比分析:")
        
        # MiniLM vs SciBERT 对比
        if "dense" in runs and "scibert" in runs:
            print("\n【MiniLM vs SciBERT】纯检索对比:")
            metrics = ["MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50"]
            for metric in metrics:
                base_val = results_table["dense"][metric]
                current_val = results_table["scibert"][metric]
                diff = current_val - base_val
                pct = (diff / base_val * 100) if base_val > 0 else 0
                print(f"  {metric:12s}: MiniLM={base_val:.4f}, SciBERT={current_val:.4f}, "
                      f"diff={diff:+.4f} ({pct:+.2f}%)")
        
        # MiniLM+BM25 vs SciBERT+BM25 对比
        if "dense_bm25" in runs and "scibert_bm25" in runs:
            print("\n【MiniLM+BM25 vs SciBERT+BM25】融合检索对比:")
            metrics = ["MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50"]
            for metric in metrics:
                base_val = results_table["dense_bm25"][metric]
                current_val = results_table["scibert_bm25"][metric]
                diff = current_val - base_val
                pct = (diff / base_val * 100) if base_val > 0 else 0
                print(f"  {metric:12s}: MiniLM+BM25={base_val:.4f}, SciBERT+BM25={current_val:.4f}, "
                      f"diff={diff:+.4f} ({pct:+.2f}%)")
        
        # MiniLM+Rerank vs SciBERT+Rerank 对比
        if "rerank" in runs and "scibert_rerank" in runs:
            print("\n【MiniLM+Rerank vs SciBERT+Rerank】重排对比:")
            metrics = ["MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50"]
            for metric in metrics:
                base_val = results_table["rerank"][metric]
                current_val = results_table["scibert_rerank"][metric]
                diff = current_val - base_val
                pct = (diff / base_val * 100) if base_val > 0 else 0
                print(f"  {metric:12s}: MiniLM+Rerank={base_val:.4f}, SciBERT+Rerank={current_val:.4f}, "
                      f"diff={diff:+.4f} ({pct:+.2f}%)")
        
        # MiniLM 内部对比
        if "dense" in runs and "dense_bm25" in runs and "rerank" in runs:
            print("\n【MiniLM 内部对比】:")
            base_system = "dense"
            for system_name in ["dense_bm25", "rerank"]:
                if system_name in runs:
                    print(f"\n{base_system} vs {system_name}:")
                    metrics = ["MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50"]
                    for metric in metrics:
                        base_val = results_table[base_system][metric]
                        current_val = results_table[system_name][metric]
                        diff = current_val - base_val
                        pct = (diff / base_val * 100) if base_val > 0 else 0
                        print(f"  {metric:12s}: {base_system}={base_val:.4f}, {system_name}={current_val:.4f}, "
                              f"diff={diff:+.4f} ({pct:+.2f}%)")
        
        # SciBERT 内部对比
        if "scibert" in runs and "scibert_bm25" in runs and "scibert_rerank" in runs:
            print("\n【SciBERT 内部对比】:")
            base_system = "scibert"
            for system_name in ["scibert_bm25", "scibert_rerank"]:
                if system_name in runs:
                    print(f"\n{base_system} vs {system_name}:")
                    metrics = ["MRR@10", "Success@1", "Success@5", "Success@10", "Recall@50", "NDCG@10", "NDCG@50"]
                    for metric in metrics:
                        base_val = results_table[base_system][metric]
                        current_val = results_table[system_name][metric]
                        diff = current_val - base_val
                        pct = (diff / base_val * 100) if base_val > 0 else 0
                        print(f"  {metric:12s}: {base_system}={base_val:.4f}, {system_name}={current_val:.4f}, "
                              f"diff={diff:+.4f} ({pct:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("指标说明:")
    print("  MRR@10      : 平均倒数排名，rel>0视为相关")
    print("  Success@1   : 第1位有相关文档的查询比例")
    print("  Success@5   : 前5位有相关文档的查询比例")
    print("  Success@10  : 前10位有相关文档的查询比例")
    print("  Recall@50   : 前50位中检索到的相关文档比例")
    print("  NDCG@10     : 归一化折损累积增益（使用3/2/1/0原始分值）")
    print("  NDCG@50     : 前50位的NDCG（使用3/2/1/0原始分值）")
    print("=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
