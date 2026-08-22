"""
app.py — Polished Streamlit UI for Meridian Supply Chain RAG
"""

import streamlit as st
from pathlib import Path
import os
import shutil
from dotenv import load_dotenv

from ingest import load_and_split, create_vectorstore, get_vectorstore
from rag import ask_question

load_dotenv()

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian Supply Chain Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        color: #5a6a7a;
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.7rem 1rem;
        margin: 0.35rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
    .success-box {
        background: #ecfdf5;
        border: 1px solid #10b981;
        padding: 0.85rem 1.1rem;
        border-radius: 8px;
        color: #065f46;
        margin: 0.8rem 0;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Meridian Components")
    st.caption("Supply Chain RAG Assistant")
    st.divider()

    chroma_path = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
    index_exists = chroma_path.exists() and any(chroma_path.iterdir())

    if index_exists:
        st.success("✅ Index ready")
        try:
            vs = get_vectorstore()
            count = vs._collection.count()
            st.metric("Chunks stored", count)
        except Exception:
            st.metric("Chunks stored", "—")
    else:
        st.warning("⚠️ No index found")
        st.caption("Upload PDFs and click **Index Documents**")

    st.divider()
    st.markdown("**Retrieval settings**")
    top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=3,
        max_value=8,
        value=6,
        help="Use 5–6 for cross-document questions",
    )

    st.divider()
    if st.button("🗑️ Clear Index", use_container_width=True):
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("Models")
    st.code("Embedding: text-embedding-3-small\nLLM: gpt-4o (temp 0.1)", language=None)

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown(
    '<p class="main-header">Meridian Supply Chain Assistant</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Ask questions across the Q1 Performance Review and Procurement Policy Handbook</p>',
    unsafe_allow_html=True,
)

# ── 1. Upload ──
with st.expander("① Upload Documents", expanded=not index_exists):
    uploaded_files = st.file_uploader(
        "Drop the Meridian PDFs here (or any additional PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Required files are already in /data. You can re-upload or add more.",
    )

    if uploaded_files:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        saved_paths = []
        for f in uploaded_files:
            path = data_dir / f.name
            path.write_bytes(f.getbuffer())
            saved_paths.append(str(path))
        st.session_state["pdf_paths"] = saved_paths
        st.success(
            f"Saved {len(saved_paths)} file(s): "
            f"{[Path(p).name for p in saved_paths]}"
        )

# Pre-load the two official PDFs if nothing was uploaded yet
if "pdf_paths" not in st.session_state:
    default_pdfs = list(Path("data").glob("*.pdf"))
    if default_pdfs:
        st.session_state["pdf_paths"] = [str(p) for p in default_pdfs]

# ── 2. Index ──
col1, col2 = st.columns([1, 3])
with col1:
    index_btn = st.button(
        "② Index Documents", type="primary", use_container_width=True
    )

if index_btn:
    paths = st.session_state.get("pdf_paths", [])
    if not paths:
        st.error("No PDFs found. Please upload files first.")
    else:
        with st.spinner("Reading → Chunking → Embedding → Storing in ChromaDB…"):
            try:
                chunks = load_and_split(paths)
                create_vectorstore(chunks)
                st.session_state["last_chunk_count"] = len(chunks)
                st.markdown(
                    f'<div class="success-box">'
                    f"✅ <b>{len(paths)} files</b> processed → "
                    f"<b>{len(chunks)} chunks</b> stored. "
                    f"Index is now persistent on disk."
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.rerun()
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ── 3. Ask ──
st.markdown("---")
st.markdown("### ③ Ask a Question")

example_questions = [
    "Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
    "How many line stoppages happened in Q1, what was the total downtime, and what caused them?",
    "What is the approval authority for a purchase order worth ₹1.4 crore?",
    "What are the four supplier classification categories, and what qualifies a supplier as Critical?",
    "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?",
    "The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?",
    "Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?",
    "Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?",
    "Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?",
    "What is the annual salary of the Head of Procurement?",  # deliberate trap
]

selected = st.selectbox(
    "Quick examples (including the trap question):",
    [""] + example_questions,
    format_func=lambda x: "— Select an example —" if x == "" else x,
)

question = st.text_area(
    "Your question",
    value=selected if selected else "",
    height=90,
    placeholder="e.g. What happens if on-time delivery falls below 85% for two consecutive quarters?",
)

ask_col, _ = st.columns([1, 4])
with ask_col:
    ask_btn = st.button("Ask →", type="primary", use_container_width=True)

if ask_btn and question.strip():
    if not index_exists:
        st.warning("Please index the documents first.")
    else:
        with st.spinner("Retrieving relevant chunks and generating answer…"):
            try:
                vs = get_vectorstore()
                answer, sources = ask_question(vs, question.strip(), top_k=top_k)

                st.markdown("#### Answer")
                st.markdown(answer)

                if sources:
                    st.markdown("#### Sources used")
                    for s in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>{s["file"]}</b> &nbsp;·&nbsp; Page {s["page"]}'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No source chunks returned.")
            except Exception as e:
                st.error(f"Error while answering: {e}")

elif ask_btn:
    st.warning("Please type or select a question.")

# Footer
st.markdown("---")
st.caption(
    "Built for Assignment 2 · Answers are grounded only in the uploaded documents · "
    "Never invents information"
)
