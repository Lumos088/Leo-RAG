import argparse
import csv
import io
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from scripts.retrieval.eval import load_qrels, load_run, read_text_lines


SYSTEMS = ("dense", "dense_bm25", "rerank")
TOPICS_PATH = os.path.join(PROJECT_DIR, "topics.csv")
QRELS_PATH = os.path.join(PROJECT_DIR, "qrels.csv")
META_PATH = os.path.join(PROJECT_DIR, "vector_db", "kb_meta.json")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data", "annotation", "annotation_pool.csv")


def load_topics(path):
    lines, _encoding = read_text_lines(path)
    reader = csv.DictReader(io.StringIO("".join(lines)))
    return list(reader)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return {item["id"]: item for item in json.load(f)}


def build_pool(top_n):
    topics = load_topics(TOPICS_PATH)
    metadata = load_metadata(META_PATH)
    _binary, graded, _stats = load_qrels(QRELS_PATH)
    runs = {}

    for system in SYSTEMS:
        path = os.path.join(RUNS_DIR, f"{system}.run")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 run 文件: {path}")
        runs[system] = load_run(path)

    rows = []
    for topic in topics:
        qid = topic["qid"]
        candidates = {}

        for system, run in runs.items():
            results = sorted(run.get(qid, []), key=lambda item: item["rank"])
            for result in results[:top_n]:
                doc_id = result["doc_id"]
                candidate = candidates.setdefault(
                    doc_id,
                    {
                        "ranks": {},
                        "scores": {},
                    },
                )
                candidate["ranks"][system] = result["rank"]
                candidate["scores"][system] = result["score"]

        ordered = sorted(
            candidates.items(),
            key=lambda item: (
                min(item[1]["ranks"].values()),
                -len(item[1]["ranks"]),
                item[0],
            ),
        )

        for doc_id, candidate in ordered:
            meta = metadata.get(doc_id, {})
            existing_rel = graded.get(qid, {}).get(doc_id, "")
            rows.append(
                {
                    "qid": qid,
                    "query": topic.get("query", ""),
                    "type": topic.get("type", ""),
                    "doc_id": doc_id,
                    "subject": meta.get("subject", ""),
                    "chunk_file": meta.get("chunk_file", ""),
                    "source_systems": "/".join(system for system in SYSTEMS if system in candidate["ranks"]),
                    "dense_rank": candidate["ranks"].get("dense", ""),
                    "dense_bm25_rank": candidate["ranks"].get("dense_bm25", ""),
                    "rerank_rank": candidate["ranks"].get("rerank", ""),
                    "dense_score": candidate["scores"].get("dense", ""),
                    "dense_bm25_score": candidate["scores"].get("dense_bm25", ""),
                    "rerank_score": candidate["scores"].get("rerank", ""),
                    "text": meta.get("text", ""),
                    "rel": existing_rel,
                    "review_status": "existing" if existing_rel != "" else "pending",
                    "review_note": "",
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="合并多个检索系统的 Top-N，生成相关性标注候选池")
    parser.add_argument("--top-n", type=int, default=10, help="每个系统、每个查询进入候选池的数量")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    args = parser.parse_args()

    if args.top_n < 1:
        parser.error("--top-n 必须大于 0")

    rows = build_pool(args.top_n)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pending_rows = [row for row in rows if row["review_status"] == "pending"]
    pending_queries = {row["qid"] for row in pending_rows}
    print(f"候选池已生成: {args.output}")
    print(f"候选总数: {len(rows)}")
    print(f"待标注候选: {len(pending_rows)}")
    print(f"涉及待标注查询: {len(pending_queries)}")


if __name__ == "__main__":
    main()
