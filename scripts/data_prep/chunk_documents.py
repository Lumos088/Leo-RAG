# chunk_documents.py
# 用法：先运行 clean_documents.py 生成 cleaned_documents.json，再运行本脚本：
#   python chunk_documents.py

import os
import json
import re
from typing import List, Dict, Any

# ========= 路径配置 =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

CLEANED_PATH = os.path.join(CLEANED_DIR, "cn_cleaned_documents.json")
CHUNKS_PATH = os.path.join(CHUNKS_DIR, "cn_chunks.json")


# ========= chunking 超参数 =========
TARGET_MAX_CHARS = 900      # 目标长度
HARD_MAX_CHARS = 1200       # 硬上限
MIN_CHUNK_CHARS = 200       # 认为“太短”的阈值
OVERLAP_SENTENCES = 1       # 相邻 chunk 之间重叠的句数
SINGLE_SENT_MAX = 400       # 单行过长时的二次切分长度


# ========= 标题模式：遇到这些行就作为分段信号 =========
HEADING_PATTERNS = [
    r"^Chapter\s+\d+",            # Chapter 1
    r"^CHAPTER\s+\d+",
    r"^\d+(\.\d+)+\s+\S+",        # 1.2.3 Some Title
    r"^\d+\.\s+\S+",              # 1. Introduction
    r"^第[一二三四五六七八九十百千]+\s*章",  # 第三章
    r"^第\s*\d+\s*章",            # 第3章
]


def is_heading(line: str) -> bool:
    """判断某一行是否为章节标题。"""
    s = line.strip()
    if not s:
        return False
    for pat in HEADING_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def split_long_sentence(s: str, max_len: int = SINGLE_SENT_MAX) -> List[str]:
    """
    对特别长的一行再做一次切分：
    1) 先按中英文句号/问号/感叹号/点号拆块；
    2) 仍然过长的块再按固定长度硬切。
    """
    s = s.strip()
    if len(s) <= max_len:
        return [s]

    parts = re.split(r"([。！？!?\.])", s)
    chunks = []
    buf = ""

    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) <= max_len:
            buf += p
        else:
            if buf:
                chunks.append(buf.strip())
            buf = p
    if buf:
        chunks.append(buf.strip())

    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            for i in range(0, len(c), max_len):
                piece = c[i:i + max_len].strip()
                if piece:
                    final.append(piece)
    return final


def chunk_one_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    对单篇文档进行 chunking：
    - 输入：清洗后的 doc = {id, text, source, subject, lang}
    - 输出：若干个 chunk dict，id 形如 "docid_chunk_0"
    """
    text = doc.get("text", "") or ""
    if not text.strip():
        return []

    lang = doc.get("lang", "en")

    # 1) 先按行拆句（clean_en / clean_zh 已经是一句一行）
    raw_lines = text.split("\n")
    sentences: List[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > SINGLE_SENT_MAX:
            sentences.extend(split_long_sentence(line, max_len=SINGLE_SENT_MAX))
        else:
            sentences.append(line)

    chunks_text: List[str] = []
    current_sents: List[str] = []
    current_len = 0
    overlap_buffer: List[str] = []

    def flush_chunk():
        """把 current_sents 收尾成一个 chunk_text。"""
        nonlocal current_sents, current_len, overlap_buffer, chunks_text
        if not current_sents:
            return
        chunk_text = "\n".join(current_sents).strip()
        if not chunk_text:
            current_sents = []
            current_len = 0
            overlap_buffer = []
            return

        chunks_text.append(chunk_text)

        # 更新 overlap 缓存（最后 N 句）
        if OVERLAP_SENTENCES > 0 and len(current_sents) >= OVERLAP_SENTENCES:
            overlap_buffer = current_sents[-OVERLAP_SENTENCES:]
        else:
            overlap_buffer = []

        current_sents = []
        current_len = 0

    # 2) 主循环：按长度 + 标题规则构建 chunk 列表（text 形式）
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue

        # 标题：作为新 chunk 的开头，而不是单独成块
        if is_heading(s):
            # 先把前面正在累积的块收尾
            flush_chunk()
            # 开启一个新的块，把标题作为第一句
            current_sents = [s]
            current_len = len(s) + 1
            continue

        s_len = len(s)

        # 情况 1：当前块 + 本句 <= 目标长度，直接加入
        if current_len + s_len + 1 <= TARGET_MAX_CHARS:
            current_sents.append(s)
            current_len += s_len + 1
            continue

        # 情况 2：当前块仍然太短（< MIN），允许适度超出，尽量避免碎片
        if current_len < MIN_CHUNK_CHARS and current_len + s_len + 1 <= HARD_MAX_CHARS:
            current_sents.append(s)
            current_len += s_len + 1
            continue

        # 情况 3：正常收尾当前块，然后基于 overlap 开启新块
        flush_chunk()

        if overlap_buffer:
            current_sents = overlap_buffer.copy()
            current_len = sum(len(x) + 1 for x in current_sents)
        else:
            current_sents = []
            current_len = 0

        current_sents.append(s)
        current_len += s_len + 1

    # 3) 末尾残余块
    flush_chunk()

    # 4) 第二轮：合并过短 chunk
    merged_texts: List[str] = []
    i = 0
    n = len(chunks_text)

    while i < n:
        cur = chunks_text[i]
        cur_len = len(cur)

        # 最后一个或者已经不短：直接收
        if cur_len >= MIN_CHUNK_CHARS or i == n - 1:
            merged_texts.append(cur)
            i += 1
            continue

        # 当前块过短，尝试和下一个合并
        nxt = chunks_text[i + 1]
        if cur_len + 1 + len(nxt) <= HARD_MAX_CHARS:
            # 合并进下一个，把新的内容写回 chunks_text[i+1]，然后跳过当前
            chunks_text[i + 1] = cur + "\n" + nxt
            i += 1
        else:
            # 合并会超限，只能单独保留
            merged_texts.append(cur)
            i += 1

    # 5) 包装成带 meta 的最终 chunk 结构
    result_chunks: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(merged_texts):
        result_chunks.append(
            {
                "id": f"{doc['id']}_chunk_{idx}",
                "text": chunk_text,
                "source": doc["source"],
                "subject": doc["subject"],
                "lang": lang,
            }
        )
    return result_chunks


def basic_chunk_stats(chunks: List[Dict[str, Any]]):
    """打印 chunk 长度分布，检查是否合理。"""
    lengths = [len(c.get("text", "") or "") for c in chunks]
    lengths = [L for L in lengths if L > 0]
    if not lengths:
        print("⚠️ 没有可统计的 chunk 长度。")
        return

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def pct(p: float) -> int:
        return lengths_sorted[int(p * (n - 1))]

    print("\n====== Chunk 长度统计（按字符数）======")
    print(f"chunk 总数：{n}")
    print(f"最短：{lengths_sorted[0]}")
    print(f"25% 分位：{pct(0.25)}")
    print(f"中位数：{pct(0.5)}")
    print(f"75% 分位：{pct(0.75)}")
    print(f"最长：{lengths_sorted[-1]}")

    too_short = sum(1 for L in lengths if L < MIN_CHUNK_CHARS)
    too_long = sum(1 for L in lengths if L > HARD_MAX_CHARS)
    print(f"过短 chunk（<{MIN_CHUNK_CHARS}）数量：{too_short}")
    print(f"超硬上限 chunk（>{HARD_MAX_CHARS}）数量：{too_long}")


if __name__ == "__main__":
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"找不到 {CLEANED_PATH}，请先运行 clean_documents.py 生成 cleaned_documents.json"
        )

    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        cleaned_docs = json.load(f)

    print(f"⏳ 共 {len(cleaned_docs)} 篇清洗后的文档，开始 chunking ...")

    all_chunks: List[Dict[str, Any]] = []

    for idx, doc in enumerate(cleaned_docs, start=1):
        doc_chunks = chunk_one_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  文档 {idx}/{len(cleaned_docs)}: {doc['id']} -> {len(doc_chunks)} 个 chunk")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 所有文档 chunking 完成，总计 {len(all_chunks)} 个 chunk，已写入：{CHUNKS_PATH}")

    basic_chunk_stats(all_chunks)
