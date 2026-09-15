import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict

from openai import OpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_POOL = os.path.join(PROJECT_DIR, "data", "annotation", "annotation_pool.csv")
DEFAULT_MODEL = "deepseek-chat"
OUTPUT_COLUMNS = ("auto_rel", "auto_confidence", "auto_reason")


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for column in OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
        for row in rows:
            row.setdefault(column, "")
    return rows, fieldnames


def save_rows(path, rows, fieldnames):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def parse_json_array(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("模型响应中没有 JSON 数组")
    data = json.loads(match.group())
    if not isinstance(data, list):
        raise ValueError("模型响应不是 JSON 数组")
    return data


def label_batch(client, model, query, candidates, max_chars):
    documents = [
        {
            "doc_id": row["doc_id"],
            "subject": row.get("subject", ""),
            "text": row.get("text", "")[:max_chars],
        }
        for row in candidates
    ]
    system_prompt = (
        "你是计算机课程检索评测员。只判断候选片段对问题的直接回答价值，不补充外部知识。"
        "相关性标准：3=完整回答核心问题；2=包含关键答案但不完整；"
        "1=主题相关但不能直接回答；0=无关。"
        "必须返回严格 JSON 数组，每项包含 doc_id、rel、confidence、reason。"
        "rel 只能是0/1/2/3；confidence 是0到1；reason不超过40个汉字。"
    )
    user_prompt = f"问题：{query}\n候选片段：\n{json.dumps(documents, ensure_ascii=False)}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return parse_json_array(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="使用 DeepSeek 对候选池进行可断点续跑的相关性初标")
    parser.add_argument("--pool", default=DEFAULT_POOL, help="候选池 CSV")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1000, help="每条候选发送给模型的最大字符数")
    parser.add_argument("--limit-queries", type=int, default=0, help="仅处理前 N 个查询；0 表示全部")
    parser.add_argument("--delay", type=float, default=0.2, help="每次请求后的等待秒数")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量")
    if args.batch_size < 1 or args.max_chars < 100:
        parser.error("--batch-size 必须大于0，--max-chars 不能小于100")

    rows, fieldnames = load_rows(args.pool)
    pending_by_query = defaultdict(list)
    for row in rows:
        if not row.get("rel", "").strip() and not row.get("auto_rel", "").strip():
            pending_by_query[row["qid"]].append(row)

    qids = sorted(pending_by_query)
    if args.limit_queries > 0:
        qids = qids[:args.limit_queries]

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    completed = 0
    failed_batches = 0

    for query_index, qid in enumerate(qids, 1):
        candidates = pending_by_query[qid]
        query = candidates[0]["query"]
        print(f"[{query_index}/{len(qids)}] {qid}: {len(candidates)} 条待初标")

        for start in range(0, len(candidates), args.batch_size):
            batch = candidates[start:start + args.batch_size]
            try:
                labels = label_batch(client, args.model, query, batch, args.max_chars)
                by_id = {item.get("doc_id"): item for item in labels if isinstance(item, dict)}
                for row in batch:
                    label = by_id.get(row["doc_id"])
                    if label is None:
                        continue
                    rel = int(label.get("rel"))
                    if rel not in (0, 1, 2, 3):
                        continue
                    row["auto_rel"] = str(rel)
                    row["auto_confidence"] = str(label.get("confidence", ""))
                    row["auto_reason"] = str(label.get("reason", ""))
                    completed += 1
                save_rows(args.pool, rows, fieldnames)
            except Exception as exc:
                failed_batches += 1
                print(f"  批次失败，已保留进度: {exc}")
            time.sleep(max(0, args.delay))

    remaining = sum(
        1 for row in rows
        if not row.get("rel", "").strip() and not row.get("auto_rel", "").strip()
    )
    print(f"初标完成: {completed} 条 | 失败批次: {failed_batches} | 剩余: {remaining} 条")


if __name__ == "__main__":
    main()
