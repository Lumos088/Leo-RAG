"""
RAG Web UI - FastAPI Backend + Streamlit Frontend

This file serves dual purpose:
1. When run with `uvicorn main:app` - FastAPI backend with REST/SSE API
2. When run with `streamlit run main.py` - Streamlit frontend UI
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Generator, Optional

import uvicorn

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
stage2_dir = os.path.dirname(current_dir)
sys.path.append(stage2_dir)

try:
    from api.rag_service import get_rag_service, RAGService
    from api.conversation_store import ConversationNotFound, ConversationStore
except ModuleNotFoundError:
    from web_ui.api.rag_service import get_rag_service, RAGService
    from web_ui.api.conversation_store import ConversationNotFound, ConversationStore
from scripts.pipeline.hybrid_retrieval import RETRIEVAL_CONFIG

# ============== FastAPI Backend ==============
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    conversation_store = ConversationStore(Path(stage2_dir) / "data" / "runtime" / "conversations.db")

    # Request/Response Models
    class ChatRequest(BaseModel):
        question: str
        mode: str = "dense"
        top_k: int = int(RETRIEVAL_CONFIG["final_top_k"])
        use_bm25: bool = True
        use_rerank: bool = True
        conversation_id: Optional[str] = None
        request_id: Optional[str] = None

    class ConversationCreate(BaseModel):
        title: Optional[str] = None

    class ConversationRename(BaseModel):
        title: str

    class ChatResponse(BaseModel):
        answer: str
        retrieval: List[dict]
        citations: List[dict]
        mode: str
        lang: str
        retrieval_query: Optional[str] = None

    class ErrorResponse(BaseModel):
        error: str

    # Create FastAPI app
    app = FastAPI(title="RAG Web UI API", version="1.0.0")

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "RAG Web UI API", "version": "1.0.0"}

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "RAG Web UI"}

    @app.get("/api/modes")
    async def get_modes():
        """Get supported retrieval modes"""
        return {
            "modes": ["dense"],
            "default": "dense"
        }

    def _conversation_or_404(conversation_id: str):
        try:
            return conversation_store.get_conversation(conversation_id)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    def _prepare_memory(service, conversation_id: str, question: str, request_id: str | None = None):
        conversation = _conversation_or_404(conversation_id)
        messages = conversation_store.list_messages(conversation_id)
        if request_id:
            messages = [item for item in messages if item.get("request_id") != request_id]
        prepared = service.memory.prepare(
            question,
            messages,
            conversation.get("summary", ""),
            conversation.get("summarized_through_message_id"),
        )
        if prepared["summary_updated"]:
            conversation_store.update_summary(
                conversation_id,
                prepared["summary"],
                prepared["summarized_through_message_id"],
            )
        return prepared

    @app.get("/api/conversations")
    async def list_conversations():
        return {"conversations": conversation_store.list_conversations()}

    @app.post("/api/conversations")
    async def create_conversation(request: ConversationCreate):
        return conversation_store.create_conversation(request.title)

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        conversation = _conversation_or_404(conversation_id)
        conversation["messages"] = conversation_store.list_messages(conversation_id)
        return conversation

    @app.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(conversation_id: str, request: ConversationRename):
        try:
            return conversation_store.rename_conversation(conversation_id, request.title)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        try:
            conversation_store.delete_conversation(conversation_id)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True}

    @app.post("/api/conversations/{conversation_id}/clear")
    async def clear_conversation(conversation_id: str):
        try:
            return conversation_store.clear_messages(conversation_id)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @app.get("/api/conversations/{conversation_id}/messages")
    async def get_conversation_messages(conversation_id: str):
        try:
            return {"messages": conversation_store.list_messages(conversation_id)}
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Non-streaming chat endpoint"""
        try:
            service = get_rag_service()
            prepared = None
            if request.conversation_id:
                if request.request_id:
                    cached = conversation_store.get_message_by_request(
                        request.conversation_id, request.request_id, "assistant"
                    )
                    if cached:
                        metadata = cached.get("metadata", {})
                        return ChatResponse(
                            answer=cached["content"], retrieval=cached["retrieval"],
                            citations=cached["citations"], mode=metadata.get("mode", request.mode),
                            lang=metadata.get("lang", "zh"),
                            retrieval_query=metadata.get("retrieval_query"),
                        )
                prepared = _prepare_memory(
                    service, request.conversation_id, request.question, request.request_id
                )
                conversation_store.add_message(
                    request.conversation_id, "user", request.question, request_id=request.request_id
                )
            result = service.query(
                request.question, request.mode,
                top_k=request.top_k, use_bm25=request.use_bm25, use_rerank=request.use_rerank,
                retrieval_question=prepared["retrieval_query"] if prepared else None,
                conversation_context=prepared["conversation_context"] if prepared else "",
            )

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            if request.conversation_id:
                conversation_store.add_message(
                    request.conversation_id, "assistant", result["answer"],
                    retrieval=result.get("retrieval", []), citations=result.get("citations", []),
                    request_id=request.request_id,
                    metadata={"mode": result["mode"], "lang": result["lang"],
                              "retrieval_query": result.get("retrieval_query"),
                              "use_bm25": request.use_bm25, "use_rerank": request.use_rerank},
                )
            return ChatResponse(**result)
        except ConversationNotFound:
            raise HTTPException(status_code=404, detail="Conversation not found")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """SSE streaming chat endpoint"""
        async def generate():
            try:
                service = get_rag_service()
                prepared = None
                if request.conversation_id:
                    _conversation_or_404(request.conversation_id)
                    if request.request_id:
                        cached = conversation_store.get_message_by_request(
                            request.conversation_id, request.request_id, "assistant"
                        )
                        if cached:
                            metadata = cached.get("metadata", {})
                            yield f"data: {json.dumps({'type': 'retrieval', 'retrieval': cached['retrieval'], 'citations': cached['citations'], 'mode': metadata.get('mode', request.mode), 'lang': metadata.get('lang', 'zh'), 'retrieval_query': metadata.get('retrieval_query')}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'chunk', 'content': cached['content'], 'replayed': True}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'done', 'answer': cached['content'], 'replayed': True}, ensure_ascii=False)}\n\n"
                            return
                    prepared = _prepare_memory(
                        service, request.conversation_id, request.question, request.request_id
                    )
                    conversation_store.add_message(
                        request.conversation_id, "user", request.question, request_id=request.request_id
                    )
                full_answer = ""
                retrieval = []
                citations = []
                lang = "zh"
                retrieval_query = prepared["retrieval_query"] if prepared else request.question
                async for chunk in service.query_stream(
                    request.question, request.mode,
                    top_k=request.top_k, use_bm25=request.use_bm25, use_rerank=request.use_rerank,
                    retrieval_question=prepared["retrieval_query"] if prepared else None,
                    conversation_context=prepared["conversation_context"] if prepared else "",
                ):
                    if chunk["type"] == "error":
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        break
                    if chunk["type"] == "retrieval":
                        retrieval = chunk.get("retrieval", [])
                        citations = chunk.get("citations", [])
                        lang = chunk.get("lang", lang)
                        retrieval_query = chunk.get("retrieval_query", retrieval_query)
                    elif chunk["type"] == "chunk":
                        full_answer += chunk.get("content", "")
                    elif chunk["type"] == "done" and request.conversation_id:
                        final_answer = chunk.get("answer") or full_answer
                        if final_answer:
                            conversation_store.add_message(
                                request.conversation_id, "assistant", final_answer,
                                retrieval=retrieval, citations=citations,
                                request_id=request.request_id,
                                metadata={"mode": request.mode, "lang": lang,
                                          "retrieval_query": retrieval_query,
                                          "use_bm25": request.use_bm25,
                                          "use_rerank": request.use_rerank},
                            )
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except ConversationNotFound:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Conversation not found'}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    def run_api_server(host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI API server"""
        uvicorn.run(app, host=host, port=port)


# ============== Streamlit Frontend ==============
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    def run_streamlit_ui():
        """Run Streamlit frontend UI"""

        # Page config
        st.set_page_config(
            page_title="RAG 问答系统",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem;
        }
        .mode-selector {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .retrieval-section {
            background-color: #fafafa;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin-top: 0.5rem;
            border-radius: 0.25rem;
        }
        .citation-badge {
            display: inline-block;
            background-color: #1f77b4;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            margin-right: 0.25rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Title
        st.markdown('<div class="main-header">🤖 RAG 智能问答系统</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Sidebar - Mode selection and settings
        with st.sidebar:
            st.header("设置")

            # Top K slider
            top_k = st.slider(
                "最大召回数量", min_value=1,
                max_value=int(RETRIEVAL_CONFIG["max_top_k"]),
                value=int(RETRIEVAL_CONFIG["final_top_k"]), step=1
            )

            use_bm25 = st.toggle("混合检索（dense+bm25）", value=True)

            # Rerank toggle (only active when hybrid is on AND in dense mode)
            use_rerank = st.toggle("启用重排（rerank）", value=True, disabled=not use_bm25)

            # Auto-disable rerank if hybrid is off
            if not use_bm25:
                use_rerank = False

            # Show effective pipeline description
            st.markdown("---")
            if use_bm25 and use_rerank:
                st.info("**当前管线**: Dense+BM25(RRF) → BGE Rerank → LLM 生成")
            elif use_bm25:
                st.info("**当前管线**: Dense+BM25(RRF) → LLM 生成")
            else:
                st.info("**当前管线**: Dense → LLM 生成")

            st.markdown("---")
            st.markdown("### 关于")
            st.markdown("""
            - 基于 DeepSeek LLM
            - 支持 Faiss 向量检索
            - 支持 BM25 关键词检索
            - 支持 BGE 重排序
            """)

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "rag_service" not in st.session_state:
            try:
                with st.spinner("正在初始化 RAG 服务..."):
                    st.session_state.rag_service = get_rag_service()
                st.success("✅ RAG 服务初始化成功")
            except Exception as e:
                st.error(f"❌ RAG 服务初始化失败: {str(e)}")
                st.session_state.rag_service = None

        # Clear chat button
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("清空对话"):
                st.session_state.messages = []

        # Helper to format score display based on retrieval mode
        def format_score(r, msg_use_bm25, msg_use_rerank):
            if msg_use_bm25 and msg_use_rerank:
                # Rerank mode: rrf + rerank
                rrf_s = r.get("rrf_score", 0.0)
                rerank_s = r.get("rerank_score", 0.0)
                return f"rrf: {rrf_s:.4f} | rerank: {rerank_s:.4f}"
            elif msg_use_bm25:
                # Hybrid mode: dense + bm25 + rrf
                dense_s = r.get("dense_score")
                bm25_s = r.get("bm25_score")
                rrf_s = r.get("rrf_score", 0.0)
                dense_str = f"{dense_s:.4f}" if dense_s is not None else "N/A"
                bm25_str = f"{bm25_s:.4f}" if bm25_s is not None else "N/A"
                return f"dense: {dense_str} | bm25: {bm25_str} | rrf: {rrf_s:.4f}"
            else:
                # Pure dense: cosine similarity
                dense_s = r.get("score", 0.0)
                return f"dense(余弦相似度): {dense_s:.4f}"

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show retrieval results for assistant messages
                if message["role"] == "assistant" and "retrieval" in message:
                    msg_use_bm25 = message.get("use_bm25", False)
                    msg_use_rerank = message.get("use_rerank", False)
                    with st.expander("📚 检索结果"):
                        for i, r in enumerate(message["retrieval"], 1):
                            st.markdown(f"**[{i}]** {format_score(r, msg_use_bm25, msg_use_rerank)}")
                            st.markdown(f"- 课程: {r.get('subject', 'N/A')}")
                            st.markdown(f"- 文件: {r.get('chunk_file', 'N/A')}")
                            st.markdown(f"- ID: {r.get('id', 'N/A')}")
                            st.markdown(f"- 文本: {r['text'][:200]}...")
                            st.markdown("---")

        # Chat input
        if prompt := st.chat_input("请输入您的问题..."):
            if not st.session_state.get("rag_service"):
                st.error("RAG 服务未初始化，请检查配置")
                st.stop()

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("正在思考和检索..."):
                    try:
                        # Get RAG service
                        service = st.session_state.rag_service

                        # Query RAG system with all parameters
                        result = service.query(
                            prompt, "dense",
                            top_k=top_k, use_bm25=use_bm25, use_rerank=use_rerank
                        )

                        if "error" in result:
                            st.error(result["error"])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"❌ {result['error']}"
                            })
                            st.stop()

                        # Display answer
                        answer = result["answer"]
                        st.markdown(answer)

                        # Show retrieval results
                        retrieval = result.get("retrieval", [])

                        # Add to session state with retrieval flags for score display
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "retrieval": retrieval,
                            "lang": result.get("lang", "zh"),
                            "use_bm25": use_bm25,
                            "use_rerank": use_bm25 and use_rerank,
                        })

                        # Render retrieval results immediately (not waiting for next rerun)
                        if retrieval:
                            with st.expander("📚 检索结果", expanded=True):
                                for i, r in enumerate(retrieval, 1):
                                    st.markdown(f"**[{i}]** {format_score(r, use_bm25, use_rerank and use_bm25)}")
                                    st.markdown(f"- 课程: {r.get('subject', 'N/A')}")
                                    st.markdown(f"- 文件: {r.get('chunk_file', 'N/A')}")
                                    st.markdown(f"- ID: {r.get('id', 'N/A')}")
                                    st.markdown(f"- 文本: {r['text'][:200]}...")
                                    st.markdown("---")

                    except Exception as e:
                        st.error(f"生成回答时出错: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ 生成回答时出错: {str(e)}"
                        })


# ============== Entry Point ==============
if __name__ == "__main__":
    # Detect if running under Streamlit
    # Streamlit sets this environment variable when running a script
    is_streamlit = "_STREAMLIT_SCRIPT_MAGIC" in globals() or "_STREAMLIT_SERVER_URL" in os.environ or "_STREAMLIT_SHARED_SERVICES" in os.environ

    # Also check if streamlit is in sys.argv
    if len(sys.argv) > 0 and "streamlit" in sys.argv[0].lower():
        is_streamlit = True

    if is_streamlit:
        # Streamlit mode
        if not STREAMLIT_AVAILABLE:
            print("Error: Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
        import streamlit_ui  # noqa: F401 - canonical Streamlit frontend
    else:
        # Direct execution - check for command line args
        import argparse

        parser = argparse.ArgumentParser(description="RAG Web UI")
        parser.add_argument("--mode", choices=["api", "ui"], help="Run mode: api for FastAPI, ui for Streamlit")
        parser.add_argument("--host", default="0.0.0.0", help="API server host")
        parser.add_argument("--port", type=int, default=8000, help="API server port")

        try:
            args = parser.parse_args()
        except:
            # Invalid args - default to API mode
            args = parser.parse_args(["--mode", "api"])

        if args.mode == "api" or args.mode is None:
            if not FASTAPI_AVAILABLE:
                print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
                sys.exit(1)
            run_api_server(host=args.host, port=args.port)
        elif args.mode == "ui":
            if not STREAMLIT_AVAILABLE:
                print("Error: Streamlit not installed. Install with: pip install streamlit")
                sys.exit(1)
            import streamlit_ui  # noqa: F401 - canonical Streamlit frontend
