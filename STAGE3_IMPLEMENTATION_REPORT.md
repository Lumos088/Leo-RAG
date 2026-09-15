# Stage 3：多会话与会话记忆实施说明

## 目标与结果

本阶段在现有 FastAPI、Streamlit 和 Stage 2 Parent-Child 检索链路上增加持久化多会话能力。检索、重排、上下文组装和引用展示保持原有逻辑。

已实现：

- 多会话的新建、切换、重命名、删除和清空。
- SQLite 持久化；服务重启后可恢复会话、消息、引用和检索结果。
- 不同会话的数据与记忆隔离。
- 最近 8 条消息作为短期记忆；更早消息压缩为持久化摘要。
- 对含有“它、这个、上述、刚才、前者、后者”等指代的追问，先改写为独立检索问题，再进入 Stage 2 检索链路。
- 原问题用于最终回答，改写后的问题只用于检索，避免改变用户表达。
- 对话历史只用于消解指代；知识性结论仍必须由本轮检索证据支持。
- 流式回答仅在收到 `done` 事件后保存助手消息；失败或中断不会保存半截答案。
- `request_id` 提供幂等保护，重试不会重复保存同一轮消息。

## 数据存储

数据库默认位置：

```text
data/runtime/conversations.db
```

数据库及 WAL 临时文件已加入 `.gitignore`。主要数据表：

- `conversations`：标题、长期摘要、摘要覆盖到的消息 ID、创建和更新时间。
- `messages`：角色、正文、检索结果、引用、运行参数、请求 ID 和状态。

## 数据链路

```text
选择会话
  → 读取长期摘要与最近 8 条消息
  → 判断当前问题是否依赖上下文
  → 必要时改写为独立检索问题
  → Stage 2 Child 检索 / Parent 聚合 / 重排 / Context Assembly
  → 原始问题 + 会话记忆 + 本轮检索证据交给 DeepSeek
  → SSE 流式返回
  → done 后原子式保存完整助手消息、引用与检索元数据
```

当历史超过 8 条消息时，超出窗口且尚未摘要的消息会并入长期摘要。DeepSeek 调用失败时，系统使用确定性的本地压缩与追问拼接作为降级方案，不阻断问答。

## API

会话接口：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/conversations` | 会话列表 |
| POST | `/api/conversations` | 新建会话 |
| GET | `/api/conversations/{id}` | 会话与消息 |
| PATCH | `/api/conversations/{id}` | 重命名 |
| DELETE | `/api/conversations/{id}` | 删除 |
| POST | `/api/conversations/{id}/clear` | 清空消息与摘要 |
| GET | `/api/conversations/{id}/messages` | 获取消息 |

`/api/chat` 与 `/api/chat/stream` 新增两个可选字段：

```json
{
  "conversation_id": "会话 ID",
  "request_id": "客户端生成的唯一请求 ID"
}
```

不传这两个字段时仍保持原有无状态 API 行为，兼容已有调用方。

## 验收标准与命令

### 自动化验收

```powershell
cd E:\RAG
python -m pytest -q
python -m py_compile .\web_ui\api\conversation_store.py .\web_ui\api\memory.py .\web_ui\api\rag_service.py .\web_ui\main.py .\web_ui\streamlit_ui.py
```

预期：项目测试全部通过；关键 Python 文件无语法错误。

覆盖项包括：

- CRUD、会话隔离、清空和删除。
- SQLite 重启恢复。
- 请求幂等与检索/引用元数据恢复。
- 追问识别、独立问题改写、长期摘要和无 LLM 降级。
- 流式失败不保存助手消息，完成后才保存，重试不重复。
- Stage 2 Parent-Child 既有回归测试。

### 运行验收

```powershell
cd E:\RAG
python .\start.py
```

打开 `http://localhost:8501`，依次验证：

1. 新建会话 A，提问“什么是死锁？”，再追问“它有哪些必要条件？”。
2. 新建会话 B，确认看不到会话 A 的消息；在 B 中提问其他课程问题。
3. 切回 A，确认原消息、答案、引用和检索结果均恢复。
4. 停止并重新运行 `start.py`，确认 A、B 仍存在。
5. 重命名 B、清空 A、删除 B，确认各操作只影响目标会话。

## 关键文件

- `web_ui/api/conversation_store.py`：SQLite 持久化与幂等写入。
- `web_ui/api/memory.py`：短期窗口、长期摘要和追问改写。
- `web_ui/api/rag_service.py`：分离原问题与检索问题，并注入受约束的会话记忆。
- `web_ui/main.py`：会话 API、聊天持久化和安全的 SSE 落库。
- `web_ui/streamlit_ui.py`：多会话操作和历史恢复。
- `tests/test_stage3_conversations.py`：Stage 3 自动化验收。

## 边界与隐私

- 当前为单机、单用户存储；尚未实现登录、用户级授权和跨设备同步。
- 会话摘要、追问改写及回答生成会把必要的会话内容发送给已配置的 DeepSeek API；本地 SQLite 不加密。
- 若后续面向多用户上线，必须先增加身份认证、`user_id` 数据隔离、访问控制、数据保留策略和数据库备份。
