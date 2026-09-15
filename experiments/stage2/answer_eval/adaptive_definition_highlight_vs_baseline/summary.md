# 自适应 Context Assembly 开发集评测

- definition：Child 高亮 + Parent
- enumeration / comparison：Parent RRF Top-5 + BGE 补位

| 指标 | 原 Chunk | Adaptive | 差值 |
|---|---:|---:|---:|
| 必备事实覆盖率 | 0.8778 | 0.8944 | +0.0167 |
| 完整事实答案率 | 0.6667 | 0.7333 | +0.0667 |
| 平均无依据主张数 | 0.0000 | 0.0000 | +0.0000 |
| 引用有效率 | 1.0000 | 1.0000 | +0.0000 |
| 清晰度（1-5） | 4.6000 | 4.8667 | +0.2667 |
| 精炼度（1-5） | 3.9333 | 4.1333 | +0.2000 |

开发集通过：**True**

门槛：`{"fact_coverage": true, "full_fact_rate": true, "citation_validity": true, "unsupported_claims": true, "critical_questions": true}`

## 逐题结果

- A004 [parent_rescue]：5/6 → 5/6；context_aware — context_aware 答案结构更清晰、冗余更少，且引用编号与上下文对应更准确，整体质量略高。
- A008 [parent_rescue]：8/8 → 8/8；tie — 两个答案均完整覆盖所有必备事实，引用有效且无无依据主张，清晰度和冗余度相当，因此并列。
- A001 [parent_rescue]：4/4 → 4/4；tie — 两个答案均完整覆盖四个必要条件，引用有效且无冗余，清晰度相同。
- A003 [parent_rescue]：6/6 → 6/6；context_aware — 两个答案均完整覆盖所有必备事实且引用有效，但 context_aware 表述更精炼、冗余度更低。
- A009 [parent_rescue]：3/3 → 3/3；context_aware — 两个答案均完整覆盖三个必备事实且引用有效，但 context_aware 的引用更精准（直接引用包含定义的上下文），且表述更精炼。
- A011 [parent_rescue]：2/4 → 2/4；tie — 两个答案均覆盖了端到端通信和流量控制两项必备事实，引用有效且无无依据主张，清晰度和冗余度相当。
- A054 [parent_rescue]：6/8 → 8/8；context_aware — context_aware 覆盖了全部 8 个必备事实，包括 baseline 缺失的 F05 和 F08，且引用有效、结构清晰。
- A020 [child_highlight_parent]：1/3 → 1/3；tie — 两个答案都仅覆盖了必备事实F01，且引用有效、无无依据主张，清晰度和冗余度相当，因此平局。
- A024 [child_highlight_parent]：4/4 → 4/4；context_aware — context_aware 覆盖全部必备事实，引用有效，结构更清晰且冗余更少。
- A058 [parent_rescue]：3/3 → 3/3；baseline — 两个答案均完整覆盖必备事实且引用有效，但 baseline 更精炼，context_aware 包含与问题无关的操作系统同步讨论，冗余度较高。
- A010 [parent_rescue]：2/2 → 2/2；baseline — 两个答案均完整覆盖两种方法且引用有效，但baseline表述更简洁精炼。
- A033 [child_highlight_parent]：5/5 → 5/5；baseline — 两者均完整覆盖所有必备事实且引用有效，但 baseline 更精炼，冗余度更低。
- A039 [child_highlight_parent]：3/4 → 3/4；context_aware — context_aware 对 O(n log n) 的定义、典型算法及与其他复杂度的对比组织更清晰，且引用均有效支持对应主张。
- A059 [parent_rescue]：5/5 → 5/5；context_aware — 两者均完整覆盖必备事实且引用有效，但 context_aware 更精炼，冗余度更低。
- A063 [parent_rescue]：5/5 → 5/5；tie — 两个答案均完整覆盖所有必备事实，引用有效且无无依据主张，清晰度和精炼度相当。
