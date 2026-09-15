from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(r"E:\RAG")
LABELS_PATH = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "required_facts.jsonl"
TOPICS_PATH = PROJECT_DIR / "topics.csv"
SCHEMA_PATH = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "schema.json"
DECISIONS_PATH = PROJECT_DIR / "data" / "stage2" / "sufficiency" / "review_decisions.csv"
BACKUP_DIR = PROJECT_DIR / "data" / "backup" / f"sufficiency_review_{datetime.now():%Y%m%d_%H%M%S}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def detect_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    raise ValueError(f"无法识别编码: {path}")


def fact(description: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {"description": description, "supporting_evidence_ids": evidence_ids}


def set_facts(label: dict[str, Any], facts: list[dict[str, Any]], minimum: list[str]) -> None:
    label["required_facts"] = [
        {
            "fact_id": f"F{index:02d}",
            "description": item["description"],
            "supporting_evidence_ids": item["supporting_evidence_ids"],
        }
        for index, item in enumerate(facts, 1)
    ]
    label["minimum_sufficient_evidence"] = [minimum]
    label["evidence_complete"] = True
    label["missing_information"] = ""
    label["review_status"] = "reviewed"
    label["evaluation_status"] = "eligible"
    label["exclusion_reason"] = ""


def exclude(label: dict[str, Any], reason: str) -> None:
    label["required_facts"] = []
    label["minimum_sufficient_evidence"] = []
    label["evidence_complete"] = False
    label["missing_information"] = reason
    label["review_status"] = "reviewed"
    label["evaluation_status"] = "excluded"
    label["exclusion_reason"] = reason


def main() -> None:
    labels = read_jsonl(LABELS_PATH)
    by_qid = {row["qid"]: row for row in labels}
    review_qids = {
        "A003", "A008", "A015", "A020", "A022", "A024", "A030", "A043",
        "A046", "A047", "A058", "A060", "A061", "A063", "A064",
    }
    if not review_qids <= set(by_qid):
        raise ValueError(f"缺少待审核题目: {sorted(review_qids - set(by_qid))}")

    BACKUP_DIR.mkdir(parents=True, exist_ok=False)
    for source in (LABELS_PATH, TOPICS_PATH, SCHEMA_PATH):
        shutil.copy2(source, BACKUP_DIR / source.name)

    original_queries = {qid: by_qid[qid]["query"] for qid in review_qids}
    decisions: dict[str, tuple[str, str]] = {}

    # 为所有记录明确评测资格；未进入本轮人工审核的记录保持 auto。
    for label in labels:
        label.setdefault("evaluation_status", "eligible")
        label.setdefault("exclusion_reason", "")

    set_facts(by_qid["A003"], [
        fact("操作系统提供进程管理与进程抽象。", ["ev_ff56efcb5b463e32d8cd", "ev_c96aa6bf487529d48b4d"]),
        fact("操作系统提供内存管理与地址空间抽象。", ["ev_ff56efcb5b463e32d8cd", "ev_c96aa6bf487529d48b4d"]),
        fact("操作系统提供文件抽象并负责文件与存储管理。", ["ev_8b1fac83837791eecf73", "ev_ff56efcb5b463e32d8cd", "ev_c96aa6bf487529d48b4d"]),
        fact("操作系统负责输入/输出（I/O）管理。", ["ev_ff56efcb5b463e32d8cd"]),
        fact("操作系统通过系统调用接口向应用程序提供服务。", ["ev_ff56efcb5b463e32d8cd", "ev_b7204828f0cecaa48ba7"]),
        fact("操作系统负责安全与保护管理。", ["ev_ff56efcb5b463e32d8cd"]),
    ], ["ev_ff56efcb5b463e32d8cd", "ev_c96aa6bf487529d48b4d", "ev_8b1fac83837791eecf73", "ev_b7204828f0cecaa48ba7"])
    decisions["A003"] = ("收缩评分事实", "合并重复的抽象层与管理层表述，保留六类核心功能。")

    by_qid["A008"]["query"] = "教材中介绍的主要页面置换算法有哪些"
    set_facts(by_qid["A008"], [
        fact("最优页面置换算法（OPT）置换未来最长时间不会被访问的页面，主要作为理论基准。", ["ev_f94698249082b4b24306"]),
        fact("最近未使用算法（NRU）依据访问位和修改位对页面分类并选择淘汰页面。", ["ev_e8475d6c43216ce9bfb4"]),
        fact("先进先出算法（FIFO）淘汰最早进入内存的页面。", ["ev_5a139f09709f3756e2a5", "ev_ce1b48fdd0fcf7d071fa"]),
        fact("第二次机会与时钟算法利用访问位改进 FIFO，时钟算法以环形结构实现。", ["ev_5a139f09709f3756e2a5", "ev_802b4ee12de35a45f0ba"]),
        fact("最近最少使用算法（LRU）淘汰最长时间未被使用的页面。", ["ev_7d4291d42e48e1bcb788"]),
        fact("NFU 与老化算法通过访问计数近似 LRU，老化算法使用移位计数器保留近期信息。", ["ev_ef9bc4ad12ea817b74ef", "ev_ce1b48fdd0fcf7d071fa"]),
        fact("工作集算法依据进程当前工作集选择淘汰页面。", ["ev_803e5cd3083a4e076ca8", "ev_ce1b48fdd0fcf7d071fa"]),
        fact("WSClock 将工作集思想与时钟算法结合。", ["ev_ca4e529bde7117781533", "ev_ce1b48fdd0fcf7d071fa"]),
    ], ["ev_f94698249082b4b24306", "ev_e8475d6c43216ce9bfb4", "ev_5a139f09709f3756e2a5", "ev_802b4ee12de35a45f0ba", "ev_7d4291d42e48e1bcb788", "ev_ef9bc4ad12ea817b74ef", "ev_803e5cd3083a4e076ca8", "ev_ca4e529bde7117781533"])
    decisions["A008"] = ("改写问题并收缩事实", "将开放的“常见类型”限定为教材范围，删除边界不清的随机策略。")

    by_qid["A015"]["query"] = "图的两种基本遍历算法是什么"
    set_facts(by_qid["A015"], [
        fact("图的基本遍历算法之一是深度优先搜索（DFS）。", ["ev_407c0501659a940e8c5a", "ev_525ae0a2f928c685d153"]),
        fact("图的基本遍历算法之一是广度优先搜索（BFS）。", ["ev_407c0501659a940e8c5a", "ev_525ae0a2f928c685d153"]),
    ], ["ev_407c0501659a940e8c5a"])
    decisions["A015"] = ("改写问题", "教材只明确给出 DFS 和 BFS，将“3种”纠正为“两种基本遍历算法”。")

    set_facts(by_qid["A020"], [
        fact("SYN Flood 是一种利用 TCP 连接管理机制的经典拒绝服务（DoS）攻击。", ["ev_74e9c9fd30d45fb18e0e"]),
        fact("攻击会在目标主机中建立大量半开或全开的 TCP 连接。", ["ev_0e271f6770a91754d805"]),
        fact("大量异常连接会使目标停止接受合法连接。", ["ev_0e271f6770a91754d805"]),
    ], ["ev_74e9c9fd30d45fb18e0e", "ev_0e271f6770a91754d805"])
    decisions["A020"] = ("收缩评分事实", "删除现有材料未直接说明的 SYN 队列耗尽细节。")

    exclude(by_qid["A022"], "问题未指明具体 Tarjan 算法，现有材料只有论文引用，不能支持算法定义。")
    decisions["A022"] = ("排除评测", "问题含义不唯一，现有材料不能定义任何具体 Tarjan 算法。")

    set_facts(by_qid["A024"], [
        fact("滑动窗口协议控制数据的发送、接收、确认与重传过程。", ["ev_18adeaea7dc94aa191ac"]),
        fact("发送窗口和接收窗口分别限定当前允许发送和接收的连续序号范围。", ["ev_ec40e3ee6f631bd9656e", "ev_f62ea2ee211f81da2430"]),
        fact("收到确认或数据后，窗口会沿序号空间向前滑动。", ["ev_ec40e3ee6f631bd9656e", "ev_9a024989bae963e37fff"]),
        fact("滑动窗口可实现流量控制，使发送速率适应接收能力。", ["ev_8c2fa9fe31794cf597ed", "ev_f8157ff88a337beff6a4"]),
    ], ["ev_18adeaea7dc94aa191ac", "ev_ec40e3ee6f631bd9656e", "ev_9a024989bae963e37fff", "ev_8c2fa9fe31794cf597ed"])
    decisions["A024"] = ("收缩评分事实", "合并发送窗、接收窗和滑动动作，去掉 TCP 专属细节。")

    exclude(by_qid["A030"], "现有材料描述 Linux 打开文件表和 i 节点，但没有给出文件控制块（FCB）的定义及字段。")
    decisions["A030"] = ("排除评测", "现有材料不能直接回答 FCB 定义。")

    exclude(by_qid["A043"], "现有材料不足以支持该问题的完整答案。")
    decisions["A043"] = ("排除评测", "现有证据不足，且本轮不补充资料。")

    exclude(by_qid["A046"], "现有材料提到相关调度算法，但未给出最短剩余时间优先（SRTF）的完整定义。")
    decisions["A046"] = ("排除评测", "现有证据不足以定义 SRTF。")

    by_qid["A047"]["review_status"] = "reviewed"
    by_qid["A047"]["evaluation_status"] = "eligible"
    by_qid["A047"]["exclusion_reason"] = ""
    decisions["A047"] = ("确认保留", "单事实问题定义清晰，证据充分，多组最小证据均可独立支撑。")

    by_qid["A058"]["query"] = "数据通信中同步传输与异步传输的区别是什么"
    set_facts(by_qid["A058"], [
        fact("同步传输要求收发双方建立时钟同步，并连续发送和接收同步比特流。", ["ev_56ebba142283b95548d5"]),
        fact("异步传输以字符或帧为单位发送，每个帧的开始时间可以是任意的。", ["ev_56ebba142283b95548d5"]),
        fact("同步传输的数据率较高，但实现代价也较高。", ["ev_56ebba142283b95548d5"]),
    ], ["ev_56ebba142283b95548d5"])
    decisions["A058"] = ("限定问题范围", "限定为数据通信，删除操作系统调用和时分复用两个不同语境。")

    set_facts(by_qid["A060"], [
        fact("作业调度决定哪些作业从外存调入内存运行。", ["ev_e2c264b6ab498750058e"]),
        fact("进程调度从就绪进程中选择获得 CPU 的进程。", ["ev_668a0f19eaf50b7186d5", "ev_fa6aa95c65ec1e2f8d13"]),
    ], ["ev_e2c264b6ab498750058e", "ev_668a0f19eaf50b7186d5"])
    decisions["A060"] = ("收缩评分事实", "只比较调度对象与作用，删除材料中需推断的频率和系统类型。")

    set_facts(by_qid["A061"], [
        fact("快速排序和归并排序都采用分治策略，但工作分布不同：归并排序重在合并，快速排序重在划分。", ["ev_4b3e99f84abbe64e1608"]),
        fact("归并排序最坏时间复杂度为 O(nlogn)；快速排序平均为 O(nlogn)，最坏为 O(n²)。", ["ev_2401073e6399464f55a9", "ev_f93755030d39c3b21cc3"]),
        fact("教材所述实现中，归并排序的归并过程需要额外存储空间，快速排序不需要归并用临时空间。", ["ev_2401073e6399464f55a9"]),
        fact("快速排序由于划分过程中的元素交换而不稳定。", ["ev_f93755030d39c3b21cc3"]),
    ], ["ev_4b3e99f84abbe64e1608", "ev_2401073e6399464f55a9", "ev_f93755030d39c3b21cc3"])
    decisions["A061"] = ("收缩并纠正评分事实", "删除无证据的归并稳定性和链表适用性，只保留材料明确支持的差异。")

    set_facts(by_qid["A063"], [
        fact("数组使用连续存储，链表结点可分散存储并通过指针连接。", ["ev_869a12abc14c7d6173ae", "ev_fdcae8e959d9dd03b52e"]),
        fact("数组可按下标 O(1) 随机访问，链表按位置访问通常需要 O(n) 遍历。", ["ev_80cf77155ef78af2fd4f", "ev_f02950cf224ab30f6a64"]),
        fact("确定位置后，链表插入删除只需修改指针；数组插入删除通常需要移动元素。", ["ev_1954f3b5b9909e03179a", "ev_80cf77155ef78af2fd4f"]),
        fact("数组存储密度较高，链表需要额外的指针空间。", ["ev_f02950cf224ab30f6a64", "ev_fdcae8e959d9dd03b52e"]),
        fact("数组适合随机访问较多的场景，链表适合频繁插入和删除的场景。", ["ev_80cf77155ef78af2fd4f"]),
    ], ["ev_869a12abc14c7d6173ae", "ev_fdcae8e959d9dd03b52e", "ev_80cf77155ef78af2fd4f", "ev_1954f3b5b9909e03179a", "ev_f02950cf224ab30f6a64"])
    decisions["A063"] = ("合并评分事实", "把十个细碎事实合并为存储、访问、修改、空间和适用场景五个维度。")

    set_facts(by_qid["A064"], [
        fact("IPv4 地址为 32 位，IPv6 地址为 128 位。", ["ev_addce0099049e150acdc", "ev_dfd696c14addaf32f236", "ev_d4eb4600d6d61d095b4b"]),
        fact("IPv6 使用固定基本首部和扩展首部，IPv4 首部可包含可变选项。", ["ev_addce0099049e150acdc", "ev_626ad187b88c1fe1c362"]),
        fact("IPv6 仅允许源结点分片，IPv4 可由中间路由器分片。", ["ev_dfd696c14addaf32f236"]),
        fact("IPv6 没有广播地址类型，使用多播等地址类型；IPv4 支持广播。", ["ev_64a492e3909e864da4d4", "ev_d4eb4600d6d61d095b4b"]),
        fact("IPv6 支持自动配置。", ["ev_addce0099049e150acdc", "ev_dfd696c14addaf32f236"]),
        fact("IPv6 与 IPv4 首部不兼容，过渡可采用双协议栈或隧道等机制。", ["ev_62167bb3408cc28e615b", "ev_64a492e3909e864da4d4"]),
    ], ["ev_addce0099049e150acdc", "ev_626ad187b88c1fe1c362", "ev_dfd696c14addaf32f236", "ev_64a492e3909e864da4d4", "ev_d4eb4600d6d61d095b4b", "ev_62167bb3408cc28e615b"])
    decisions["A064"] = ("收缩并纠正评分事实", "删除安全性、资源预分配、PDU 名称等边缘或易争议表述，保留六项核心差异。")

    # 同步 topics.csv 中三道被改写的问题，保持原列和原编码。
    topic_encoding = detect_encoding(TOPICS_PATH)
    with TOPICS_PATH.open("r", encoding=topic_encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        topic_rows = list(reader)
        topic_fields = list(reader.fieldnames or [])
    for row in topic_rows:
        if row.get("qid") in {"A008", "A015", "A058"}:
            row["query"] = by_qid[row["qid"]]["query"]

    LABELS_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(labels, key=lambda x: x["qid"])),
        encoding="utf-8",
    )
    with TOPICS_PATH.open("w", encoding=topic_encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=topic_fields)
        writer.writeheader()
        writer.writerows(topic_rows)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["evaluation_status"] = ["eligible", "excluded"]
    required = list(schema.get("required_fields", []))
    for field_name in ("evaluation_status", "exclusion_reason"):
        if field_name not in required:
            required.append(field_name)
    schema["required_fields"] = required
    SCHEMA_PATH.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with DECISIONS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["qid", "original_query", "final_query", "decision", "evaluation_status", "fact_count", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(review_qids):
            decision, reason = decisions[qid]
            writer.writerow({
                "qid": qid,
                "original_query": original_queries[qid],
                "final_query": by_qid[qid]["query"],
                "decision": decision,
                "evaluation_status": by_qid[qid]["evaluation_status"],
                "fact_count": len(by_qid[qid]["required_facts"]),
                "reason": reason,
            })

    print(json.dumps({
        "backup_dir": str(BACKUP_DIR),
        "reviewed": len(review_qids),
        "eligible": sum(by_qid[qid]["evaluation_status"] == "eligible" for qid in review_qids),
        "excluded": [qid for qid in sorted(review_qids) if by_qid[qid]["evaluation_status"] == "excluded"],
        "rewritten_queries": [qid for qid in sorted(review_qids) if original_queries[qid] != by_qid[qid]["query"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
