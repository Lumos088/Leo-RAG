# clean_documents.py
# 用法：先运行 load_documents.py 生成 structured_documents.json，再运行本脚本：
#   python clean_documents.py

import os
import json
import re

import spacy
import jieba

# ========== 路径配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw_documents")
CLEANED_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cleaned")
os.makedirs(CLEANED_OUTPUT_DIR, exist_ok=True)

STRUCTURED_PATH = os.path.join(RAW_OUTPUT_DIR, "all_structured_documents.json")

# 新增：配置目录与噪音模式文件路径
CONFIG_DIR = os.path.join(BASE_DIR, "config")
os.makedirs(CONFIG_DIR, exist_ok=True)
NOISE_PATTERNS_FILE = os.path.join(CONFIG_DIR, "noise_patterns.txt")

# ========== 内置默认噪音模式（兜底用） ==========
DEFAULT_NOISE_LINE_PATTERNS = [
    r"^OPERATING SYSTEMS\s*$",
    r"^THREE\s+EASY\s+PIECES\s*$",
    r"^c⃝.*$",                # c⃝2014 版权串
    r"^©.*$",                 # © 开头的版权串
    r"ARPACI[- ]DUSSEAU",    # 作者名/出版社名，通常在版权页/页眉里
    r"^图书在版编目.*$",      # 中文 CIP 开头
    r"CIP 数据",             # CIP 关键词
    r"中国版本图书馆",        # CIP 相关
    r"本书版权登记号",        # 版权页
    r"购书热线",              # 出版社广告
    r"网上购书",              # 出版社广告
    r"数字阅读",              # 出版社广告
    r"www\.[A-Za-z0-9\.-]+", # 各种网址
]


def load_noise_patterns(path: str):
    """
    从外部文件加载噪音模式：
    - 文件路径：config/noise_patterns.txt
    - 一行一个正则表达式
    - 空行与 # 开头行视为注释，自动忽略
    若文件不存在或有效行为空，则回退到 DEFAULT_NOISE_LINE_PATTERNS。
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

        if patterns:
            print(f"从 {path} 加载了 {len(patterns)} 条噪音模式")
            return patterns
        else:
            print(f"警告：{path} 中未找到有效噪音模式行，将使用内置默认模式。")
    else:
        print(f"注意：未找到噪音模式文件 {path}，将使用内置默认模式。")

    print(f"使用内置默认噪音模式 {len(DEFAULT_NOISE_LINE_PATTERNS)} 条")
    return DEFAULT_NOISE_LINE_PATTERNS


# 实际使用的噪音模式列表
NOISE_LINE_PATTERNS = load_noise_patterns(NOISE_PATTERNS_FILE)


# ========== 通用基础清洗：中英文都用这一层 ==========
def basic_clean_common(text: str) -> str:
    """
    去掉页码、多余空行、多余空格，以及典型 header/footer 垃圾行。
    同时兼容外部配置的噪音模式。
    """
    if not text:
        return ""

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 先按行拆开，逐行过滤
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        raw_line = line
        line = line.strip()

        # 空行先保留，占位，后面再压缩多空行
        if not line:
            cleaned_lines.append("")
            continue

        # 如果某行匹配任一噪音模式，就视为垃圾行
        is_noise = False
        for pat in NOISE_LINE_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                is_noise = True
                break
        if is_noise:
            continue

        cleaned_lines.append(line)

    # 再拼回文本
    text = "\n".join(cleaned_lines)

    # 连续 3 行以上空行压成 2 行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉类似 Page 123 这类页码
    text = re.sub(r"\s*Page \d+\s*", " ", text, flags=re.IGNORECASE)

    # 合并多空格 / tab
    text = re.sub(r"[ \t]+", " ", text)

    # 去掉首尾空白
    return text.strip()


# ========== 简单语言检测：看前 N 个字符里的汉字比例 ==========
def detect_lang(text: str, sample_chars: int = 1000, threshold: float = 0.1) -> str:
    """粗略判断是中文(zh)还是英文(en)，只看前 sample_chars 个非空白字符."""
    if not text:
        return "en"

    sample = "".join(ch for ch in text[:sample_chars] if not ch.isspace())
    if not sample:
        return "en"

    chinese = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    ratio = chinese / len(sample)

    return "zh" if ratio >= threshold else "en"


# ========== 英文清洗：spaCy + sentencizer + 短句安全策略 ==========
def clean_en(text: str, nlp_en) -> str:
    """
    英文文本清洗：
    1) 通用基础清洗
    2) 用 spaCy sentencizer 按句切分（如果可用）
    3) 如果没有 spaCy，使用正则表达式按句切分
    4) 丢弃极短噪声句；稍短句合并到前一句，避免丢掉有用提示
    """
    text = basic_clean_common(text)
    if not text:
        return ""

    # 按段落切分，避免一次喂入特别大的字符串
    paragraphs = re.split(r"\n{2,}", text)

    sentences = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if nlp_en is not None:
            # 使用 spaCy 按句切分
            doc = nlp_en(para)
            for sent in doc.sents:
                s = sent.text.strip()
                if not s:
                    continue

                # 极短碎片，直接丢掉（比如只有几个字符）
                if len(s) < 5:
                    continue

                # 稍短句子，尽量合并到前一句
                if len(s) < 20:
                    if sentences:
                        sentences[-1] = sentences[-1].rstrip() + " " + s
                    else:
                        sentences.append(s)
                else:
                    sentences.append(s)
        else:
            # 使用正则表达式按句切分
            # 匹配句末标点：.?!
            sent_pattern = r'[^.?!]*(?:[.!?](?=\s|$))?'
            potential_sents = re.findall(sent_pattern, para)

            for s in potential_sents:
                s = s.strip()
                if not s:
                    continue

                # 极短碎片，直接丢掉
                if len(s) < 5:
                    continue

                # 稍短句子，尽量合并到前一句
                if len(s) < 20:
                    if sentences:
                        sentences[-1] = sentences[-1].rstrip() + " " + s
                    else:
                        sentences.append(s)
                else:
                    sentences.append(s)

    return "\n".join(sentences)


# ========== 中文句子切分：基于标点 ==========
def split_zh_sentences(text: str):
    """
    基于句末标点切分中文句子，保留 。！？!? 等。
    先做基础清洗，再按标点拆分。
    """
    text = basic_clean_common(text)
    if not text:
        return []

    parts = re.split(r"(?<=[。！？!?])\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


# ========== 中文清洗：句子切分 + jieba 辅助过滤 ==========
def clean_zh(text: str, nlp_zh=None) -> str:
    """
    中文文本清洗：
    1) 通用基础清洗 + 句子级切分
    2) 用 jieba 过滤极短/无信息句子
    3) 如有 zh_core_web_sm，可过一遍以保证文本规范（可选）
    """
    sentences_raw = split_zh_sentences(text)
    cleaned_sentences = []

    for s in sentences_raw:
        s = s.strip()
        if not s:
            continue

        tokens = [tok for tok in jieba.lcut(s) if tok.strip()]
        if len(tokens) < 2:
            continue  # 单字或噪音句

        if nlp_zh is not None:
            doc = nlp_zh(s)
            s_clean = doc.text.strip()
        else:
            s_clean = s

        if s_clean:
            cleaned_sentences.append(s_clean)

    return "\n".join(cleaned_sentences)


if __name__ == "__main__":
    # 1. 检查输入文件
    if not os.path.exists(STRUCTURED_PATH):
        raise FileNotFoundError(
            f"找不到 {STRUCTURED_PATH}，请先运行 load_documents.py 生成 structured_documents.json"
        )

    with open(STRUCTURED_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print("正在加载 spaCy 模型 ...")

    # 2. 尝试加载英文模型（可选）
    try:
        nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        if "sentencizer" not in nlp_en.pipe_names:
            nlp_en.add_pipe("sentencizer")
        nlp_en.max_length = 2_000_000
        print("已加载英文spaCy模型。")
    except OSError:
        nlp_en = None
        print("警告：未找到英文spaCy模型，将使用正则表达式处理。")

    # 3. 尝试加载中文模型（可选）
    try:
        nlp_zh = spacy.load("zh_core_web_sm", disable=["parser", "ner", "textcat"])
        if "sentencizer" not in nlp_zh.pipe_names:
            nlp_zh.add_pipe("sentencizer")
        nlp_zh.max_length = 2_000_000
        print("已加载 zh_core_web_sm，用于中文文本的轻量处理。")
    except OSError:
        nlp_zh = None
        print("警告：未找到 zh_core_web_sm，将使用 正则 + jieba 处理中文。")

    print("模型加载完成，开始清洗文档 ...")

    cleaned_documents = []

    for idx, doc in enumerate(documents, start=1):
        raw_text = doc.get("text", "") or ""
        lang = detect_lang(raw_text)

        if lang == "en":
            cleaned_text = clean_en(raw_text, nlp_en)
        else:
            cleaned_text = clean_zh(raw_text, nlp_zh)

        cleaned_documents.append(
            {
                "id": doc["id"],
                "text": cleaned_text,
                "source": doc["source"],
                "subject": doc["subject"],
                "lang": lang,
            }
        )

        if idx % 10 == 0:
            print(f"  ... 已处理 {idx} 篇")

    # 4. 写出结果
    output_path = os.path.join(CLEANED_OUTPUT_DIR, "cn_cleaned_documents.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_documents, f, ensure_ascii=False, indent=2)

    print(f"文档清洗完成，共 {len(cleaned_documents)} 篇，已写入：{output_path}")
