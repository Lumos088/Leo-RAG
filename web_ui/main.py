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
from typing import Generator, Optional

import uvicorn

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
stage2_dir = os.path.dirname(current_dir)
sys.path.append(stage2_dir)

from api.rag_service import get_rag_service, RAGService

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
    # Request/Response Models
    class ChatRequest(BaseModel):
        question: str
        mode: str = "llm_retrieval"
        top_k: int = 5
        use_bm25: bool = False
        use_rerank: bool = False

    class ChatResponse(BaseModel):
        answer: str
        retrieval: List[dict]
        citations: List[dict]
        mode: str
        lang: str

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
            "modes": ["dense", "llm_retrieval"],
            "default": "llm_retrieval"
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Non-streaming chat endpoint"""
        try:
            service = get_rag_service()
            result = service.query(
                request.question, request.mode,
                top_k=request.top_k, use_bm25=request.use_bm25, use_rerank=request.use_rerank
            )

            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])

            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """SSE streaming chat endpoint"""
        async def generate():
            try:
                service = get_rag_service()
                async for chunk in service.query_stream(
                    request.question, request.mode,
                    top_k=request.top_k, use_bm25=request.use_bm25, use_rerank=request.use_rerank
                ):
                    if chunk["type"] == "error":
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        break
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
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

            # Mode selection - only 2 options
            modes = ["llm_retrieval", "dense"]
            mode_labels = {
                "llm_retrieval": "LLM 检索",
                "dense": "密集检索",
            }

            selected_mode = st.selectbox(
                "检索模式",
                modes,
                index=0,
                format_func=lambda x: mode_labels.get(x, x)
            )

            # Top K slider
            top_k = st.slider("最大召回数量", min_value=1, max_value=10, value=5, step=1)

            # Determine if toggles are enabled based on mode
            is_dense_mode = selected_mode == "dense"

            # Hybrid toggle (only active in dense mode)
            use_bm25 = st.toggle("混合检索（dense+bm25）", value=False, disabled=not is_dense_mode)

            # Rerank toggle (only active when hybrid is on AND in dense mode)
            use_rerank = st.toggle("启用重排（rerank）", value=False, disabled=not (is_dense_mode and use_bm25))

            # Auto-disable rerank if hybrid is off
            if not use_bm25:
                use_rerank = False

            # Show effective pipeline description
            st.markdown("---")
            if selected_mode == "llm_retrieval":
                st.info("**当前管线**: LLM 智能检索 → LLM 生成")
            else:
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
        def format_score(r, msg_mode, msg_use_bm25, msg_use_rerank):
            if msg_mode == "llm_retrieval":
                # LLM retrieval mode: show LLM scores + original dense score as reference
                llm_rel = r.get("llm_relevance", "N/A")
                llm_s = r.get("llm_score")
                llm_match = r.get("llm_match", "")
                llm_s_str = f"{llm_s:.2f}" if llm_s is not None else "N/A"
                dense_s = r.get("score", 0.0)
                match_str = f" | 匹配点: {llm_match}" if llm_match else ""
                return f"LLM相关性: {llm_rel} | LLM相似度: {llm_s_str} | dense: {dense_s:.4f}{match_str}"
            elif msg_use_bm25 and msg_use_rerank:
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
                    msg_mode = message.get("mode", "dense")
                    msg_use_bm25 = message.get("use_bm25", False)
                    msg_use_rerank = message.get("use_rerank", False)
                    with st.expander("📚 检索结果"):
                        for i, r in enumerate(message["retrieval"], 1):
                            st.markdown(f"**[{i}]** {format_score(r, msg_mode, msg_use_bm25, msg_use_rerank)}")
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
                            prompt, selected_mode,
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
                            "mode": selected_mode,
                            "lang": result.get("lang", "zh"),
                            "use_bm25": selected_mode == "dense" and use_bm25,
                            "use_rerank": selected_mode == "dense" and use_bm25 and use_rerank,
                        })

                        # Render retrieval results immediately (not waiting for next rerun)
                        if retrieval:
                            with st.expander("📚 检索结果", expanded=True):
                                for i, r in enumerate(retrieval, 1):
                                    st.markdown(f"**[{i}]** {format_score(r, selected_mode, use_bm25 and selected_mode == 'dense', use_rerank and selected_mode == 'dense' and use_bm25)}")
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
        run_streamlit_ui()
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
            run_streamlit_ui()
