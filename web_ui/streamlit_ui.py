"""
Streamlit UI for RAG Web UI
Separate file to avoid mode detection issues
"""
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
stage2_dir = os.path.dirname(current_dir)
sys.path.append(stage2_dir)

import streamlit as st

from api.rag_service import get_rag_service

# Page config
st.set_page_config(
    page_title="RAG Q&A System",
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
st.markdown('<div class="main-header">🤖 RAG Intelligent Q&A System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Mode selection and settings
with st.sidebar:
    st.header("Settings")

    # Mode selection - only 2 options
    modes = ["llm_retrieval", "dense"]
    mode_labels = {
        "llm_retrieval": "LLM Retrieval",
        "dense": "Dense Retrieval",
    }

    selected_mode = st.selectbox(
        "Retrieval Mode",
        modes,
        index=0,
        format_func=lambda x: mode_labels.get(x, x)
    )

    # Top K slider
    top_k = st.slider("Max Recall Count", min_value=1, max_value=10, value=5, step=1)

    # Determine if toggles are enabled based on mode
    is_dense_mode = selected_mode == "dense"

    # Hybrid toggle (only active in dense mode)
    use_bm25 = st.toggle("Hybrid Retrieval (Dense+BM25)", value=False, disabled=not is_dense_mode)

    # Rerank toggle (only active when hybrid is on AND in dense mode)
    use_rerank = st.toggle("Enable Reranking", value=False, disabled=not (is_dense_mode and use_bm25))

    # Auto-disable rerank if hybrid is off
    if not use_bm25:
        use_rerank = False

    # Show effective pipeline description
    st.markdown("---")
    if selected_mode == "llm_retrieval":
        st.info("**Current Pipeline**: LLM Retrieval → LLM Generation")
    else:
        if use_bm25 and use_rerank:
            st.info("**Current Pipeline**: Dense+BM25(RRF) → BGE Rerank → LLM Generation")
        elif use_bm25:
            st.info("**Current Pipeline**: Dense+BM25(RRF) → LLM Generation")
        else:
            st.info("**Current Pipeline**: Dense → LLM Generation")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    - Powered by DeepSeek LLM
    - FAISS vector retrieval
    - BM25 keyword retrieval
    - BGE reranking
    """)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_service" not in st.session_state:
    try:
        with st.spinner("Initializing RAG service..."):
            st.session_state.rag_service = get_rag_service()
        st.success("✅ RAG service initialized successfully")
    except Exception as e:
        st.error(f"❌ RAG service initialization failed: {str(e)}")
        st.session_state.rag_service = None

# Clear chat button
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Clear Chat"):
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
        match_str = f" | Match: {llm_match}" if llm_match else ""
        return f"LLM Rel: {llm_rel} | LLM Score: {llm_s_str} | dense: {dense_s:.4f}{match_str}"
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
        return f"dense(cosine similarity): {dense_s:.4f}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show retrieval results for assistant messages
        if message["role"] == "assistant" and "retrieval" in message:
            msg_mode = message.get("mode", "dense")
            msg_use_bm25 = message.get("use_bm25", False)
            msg_use_rerank = message.get("use_rerank", False)
            with st.expander("📚 Retrieval Results"):
                for i, r in enumerate(message["retrieval"], 1):
                    st.markdown(f"**[{i}]** {format_score(r, msg_mode, msg_use_bm25, msg_use_rerank)}")
                    st.markdown(f"- Course: {r.get('subject', 'N/A')}")
                    st.markdown(f"- File: {r.get('chunk_file', 'N/A')}")
                    st.markdown(f"- ID: {r.get('id', 'N/A')}")
                    st.markdown(f"- Text: {r['text'][:200]}...")
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Enter your question..."):
    if not st.session_state.get("rag_service"):
        st.error("RAG service not initialized, please check configuration")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving..."):
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

                # Show retrieval results in current render cycle
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
                    with st.expander("📚 Retrieval Results", expanded=True):
                        for i, r in enumerate(retrieval, 1):
                            st.markdown(f"**[{i}]** {format_score(r, selected_mode, use_bm25 and selected_mode == 'dense', use_rerank and selected_mode == 'dense' and use_bm25)}")
                            st.markdown(f"- Course: {r.get('subject', 'N/A')}")
                            st.markdown(f"- File: {r.get('chunk_file', 'N/A')}")
                            st.markdown(f"- ID: {r.get('id', 'N/A')}")
                            st.markdown(f"- Text: {r['text'][:200]}...")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error generating answer: {str(e)}"
                })
