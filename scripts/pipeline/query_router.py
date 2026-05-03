# query_router.py
import re


def detect_lang(question: str) -> str:
    """检测语言：含中文字符则视为中文"""
    for ch in question:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def detect_subject(question: str) -> str | None:
    """
    根据关键词判断课程主题
    返回：
      - "Operating System"
      - "Data Structures"
      - "Computer Networks"
      - None（不确定则不限制）
    """
    q = question.lower()

    # ===== Operating System =====
    os_keywords = [
        "process", "thread", "cpu", "scheduling",
        "context switch", "memory management",
        "paging", "virtual memory",
        "deadlock", "semaphore", "mutex",
        "kernel", "interrupt", "system call"
    ]

    # ===== Data Structures =====
    ds_keywords = [
        "stack", "queue", "heap",
        "tree", "binary tree", "bst",
        "graph", "hash", "hash table",
        "linked list", "array",
        "sorting", "search", "algorithm"
    ]

    # ===== Computer Networks =====
    cn_keywords = [
        "network", "computer network",
        "tcp", "udp", "ip",
        "http", "https", "dns",
        "osi", "osi model", "tcp/ip",
        "routing", "router", "switch",
        "congestion", "flow control",
        "packet", "latency", "bandwidth",
        "ethernet", "wifi"
    ]

    # OS 优先
    for kw in os_keywords:
        if kw in q:
            return "Operating System"

    # DS 次之
    for kw in ds_keywords:
        if kw in q:
            return "Data Structures"

    # CN 最后
    for kw in cn_keywords:
        if kw in q:
            return "Computer Networks"

    # 无法确定 → 不做过滤
    return None


def route_query(question: str):
    """
    系统级路由决策
    语言策略：
      - 英文问题：优先 en，不足再 zh
      - 中文问题：优先 zh，不足再 en
    """
    lang = detect_lang(question)

    if lang == "en":
        lang_priority = ["en", "zh"]
    else:
        lang_priority = ["zh", "en"]

    return {
        "lang_priority": lang_priority,
        "subject": detect_subject(question)
    }

