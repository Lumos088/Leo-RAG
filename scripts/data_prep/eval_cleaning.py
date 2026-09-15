# eval_cleaning.py
import os
import json
import random
import re

# ========= 路径配置 =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw_documents")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

RAW_PATH = os.path.join(RAW_DIR, "ds_structured_documents.json")
CLEAN_PATH = os.path.join(CLEAN_DIR, "ds_cleaned_documents.json")

# 新增：噪音模式配置文件路径（和 clean_documents.py 保持一致）
CONFIG_DIR = os.path.join(BASE_DIR, "config")
NOISE_PATTERNS_FILE = os.path.join(CONFIG_DIR, "noise_patterns.txt")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def basic_stats(raw_docs, clean_docs):
    print("====== 基本文档统计 ======")
    print(f"原始文档数：{len(raw_docs)}")
    print(f"清洗后文档数：{len(clean_docs)}")

    # 建一个 id -> 文本 的索引方便对照
    clean_map = {d["id"]: d for d in clean_docs}

    ratios = []
    empty_clean = 0

    for doc in raw_docs:
        doc_id = doc["id"]
        raw_text = doc.get("text", "") or ""
        clean_text = clean_map.get(doc_id, {}).get("text", "") or ""

        raw_len = len(raw_text)
        clean_len = len(clean_text)

        if clean_len == 0:
            empty_clean += 1

        if raw_len > 0:
            ratio = clean_len / raw_len
            ratios.append(ratio)

    if ratios:
        ratios_sorted = sorted(ratios)
        n = len(ratios_sorted)
        print(f"清洗后为空的文档数：{empty_clean}")
        print(f"长度比例(清洗后长度 / 原始长度)的一些统计：")
        print(f"  最小值：{ratios_sorted[0]:.3f}")
        print(f"  25%分位：{ratios_sorted[int(0.25*n)]:.3f}")
        print(f"  中位数：{ratios_sorted[int(0.5*n)]:.3f}")
        print(f"  75%分位：{ratios_sorted[int(0.75*n)]:.3f}")
        print(f"  最大值：{ratios_sorted[-1]:.3f}")
    else:
        print("⚠️ 没有有效 ratio 可统计（可能原始文本为空？）")


def load_noise_patterns(path: str):
    """
    从 config/noise_patterns.txt 中加载噪音模式：
    - 一行一个正则表达式
    - 空行和以 # 开头的注释行会被忽略
    """
    patterns = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                patterns.append(s)

    return patterns


def count_noise_patterns(raw_docs, clean_docs):
    """
    使用项目目录下 config/noise_patterns.txt 中的模式，
    统计所有噪音模式在清洗前后的“总出现次数”。
    """
    print("\n====== 噪音模式统计（清洗前 vs 清洗后） ======")

    patterns = load_noise_patterns(NOISE_PATTERNS_FILE)
    if not patterns:
        print(f"⚠️ 未在 {NOISE_PATTERNS_FILE} 中找到有效噪音模式，跳过噪音统计。")
        return

    # 拼接所有文档文本
    raw_text_all = "\n".join(d.get("text", "") or "" for d in raw_docs)
    clean_text_all = "\n".join(d.get("text", "") or "" for d in clean_docs)

    total_raw = 0
    total_clean = 0

    for pat in patterns:
        raw_count = len(re.findall(pat, raw_text_all, flags=re.IGNORECASE))
        clean_count = len(re.findall(pat, clean_text_all, flags=re.IGNORECASE))
        total_raw += raw_count
        total_clean += clean_count

    print(f"噪音模式数量：{len(patterns)} 条")
    print(f"噪音出现总次数：清洗前 {total_raw} 次 -> 清洗后 {total_clean} 次")


def sample_compare(raw_docs, clean_docs, k=3):
    """
    随机抽 k 篇文档，对比原文 & 清洗后的开头/中间/结尾部分。
    """
    print("\n====== 抽样对比（raw vs cleaned）======")

    clean_map = {d["id"]: d for d in clean_docs}
    sample_docs = random.sample(raw_docs, min(k, len(raw_docs)))

    for doc in sample_docs:
        doc_id = doc["id"]
        raw_text = doc.get("text", "") or ""
        clean_text = clean_map.get(doc_id, {}).get("text", "") or ""

        print("\n---------- 文档 ID:", doc_id, "----------")
        print("[原始文本 - 前 400 字]:")
        print(raw_text[:400].replace("\n", "\\n"))
        print("\n[清洗后文本 - 前 400 字]:")
        print(clean_text[:400].replace("\n", "\\n"))

        # 中间和末尾也各看一段（如果长度足够）
        if len(raw_text) > 1200 and len(clean_text) > 1200:
            print("\n[原始文本 - 中间 200 字]:")
            mid_r = len(raw_text) // 2
            print(raw_text[mid_r:mid_r+200].replace("\n", "\\n"))

            print("\n[清洗后文本 - 中间 200 字]:")
            mid_c = len(clean_text) // 2
            print(clean_text[mid_c:mid_c+200].replace("\n", "\\n"))

            print("\n[原始文本 - 末尾 400 字]:")
            print(raw_text[-400:].replace("\n", "\\n"))

            print("\n[清洗后文本 - 末尾 400 字]:")
            print(clean_text[-400:].replace("\n", "\\n"))


if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"找不到 {RAW_PATH}，请先运行 load_documents.py")
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"找不到 {CLEAN_PATH}，请先运行 clean_documents.py")

    raw_docs = load_json(RAW_PATH)
    clean_docs = load_json(CLEAN_PATH)

    basic_stats(raw_docs, clean_docs)
    count_noise_patterns(raw_docs, clean_docs)
    sample_compare(raw_docs, clean_docs, k=3)
