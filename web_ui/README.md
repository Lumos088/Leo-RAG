# RAG Web UI

基于 FastAPI + Streamlit 的 RAG 智能问答系统 Web 界面。

## 功能特性

- 🤖 支持 2 种核心检索模式 + 可配置管线
- 💬 友好的聊天界面
- 📚 实时检索结果展示（支持双分数显示）
- 🌏 支持中英文问答
- ⚡ 基于 DeepSeek LLM

## 检索模式

| 模式            | 说明                                    | 可配置项 |
| --------------- | --------------------------------------- | -------- |
| `llm_retrieval` | LLM 检索（推荐）- 使用 LLM 进行智能检索 | top_k    |
| `dense`         | 密集检索 - 可配置 BM25 融合与重排       | top_k, use_bm25, use_rerank |

### Dense 模式管线组合

| use_bm25 | use_rerank | 实际管线 |
|----------|------------|----------|
| false    | false      | Dense → LLM 生成 |
| true     | false      | Dense+BM25(RRF) → LLM 生成 |
| true     | true       | Dense+BM25(RRF) → BGE Rerank → LLM 生成 |

### API 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `question` | string | 必填 | 用户问题 |
| `mode` | string | `llm_retrieval` | 检索模式 |
| `top_k` | int | 5 | 返回结果数量 (1-10) |
| `use_bm25` | bool | false | 启用 BM25 混合检索 |
| `use_rerank` | bool | false | 启用 BGE 重排序 |

## 安装

```bash
cd "H:\RAG project\stage2\web_ui"
pip install -r requirements.txt
```

## 环境变量

需要设置 DeepSeek API Key:

```bash
# Windows
set DEEPSEEK_API_KEY=your_api_key_here

# Linux/Mac
export DEEPSEEK_API_KEY=your_api_key_here
```

## 使用方法

### 方式一：Streamlit 前端（推荐）

```bash
streamlit run main.py --server.port 8501
```

然后在浏览器打开 `http://localhost:8501`

### 方式二：FastAPI 后端

```bash
# 启动 API 服务
uvicorn main:app --reload --port 8000
```

访问 API 文档: `http://localhost:8000/docs`

### API 端点

#### 健康检查

```bash
curl http://localhost:8000/api/health
```

#### 获取支持的模式

```bash
curl http://localhost:8000/api/modes
```

#### 问答接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是死锁？", "mode": "llm_retrieval", "top_k": 5, "use_bm25": false, "use_rerank": false}'
```

#### 流式问答接口 (SSE)

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是死锁？", "mode": "dense", "top_k": 5, "use_bm25": true, "use_rerank": true}'
```

## 项目结构

```
web_ui/
├── api/
│   ├── __init__.py           # API 包初始化
│   └── rag_service.py       # RAG 核心服务
├── main.py                  # FastAPI + Streamlit 主入口
├── requirements.txt         # 依赖列表
└── README.md               # 本文件
```

## 依赖关系

本 Web UI 复用以下现有组件：

- `app/rag_cli.py` - RAG 逻辑提取参考
- `app/bm25_retriever.py` - BM25 检索器
- `app/reranker.py` - BGE 重排序器
- `scripts/pipeline/retriever.py` - Faiss 向量检索器
- `vector_db/kb.index` - Faiss 向量索引
- `vector_db/kb_meta.json` - 文档元数据

## 故障排除

### 1. DEEPSEEK_API_KEY 未设置错误

```
RuntimeError: ❌ 未检测到 DEEPSEEK_API_KEY 环境变量
```

**解决**: 设置环境变量 `DEEPSEEK_API_KEY`

### 2. 向量数据库文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: '.../vector_db/kb.index'
```

**解决**: 确保 `H:\RAG project\stage2\vector_db\` 目录下存在 `kb.index` 和 `kb_meta.json`

### 3. 依赖安装失败

**解决**: 使用清华源镜像安装:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 开发说明

### 添加新的检索模式

1. 在 `api/rag_service.py` 中添加新的检索方法
2. 更新 `get_modes()` API 端点
3. 在 Streamlit UI 中添加模式选项

### 自定义前端样式

编辑 `main.py` 中的 `<style>` 部分

## 许可证

与主项目保持一致
