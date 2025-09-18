# app.py
import os
import re
import json
import faiss
import faiss  # ensure faiss or faiss-cpu is installed in environment
import pickle
import base64
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# PDF / embeddings / LLM
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# optional: langchain splitter (nice but not required)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:
    _HAS_LANGCHAIN = False

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fallback; will be updated from model.get_sentence_embedding_dimension()
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]  # try pro first, fallback to flash
CHUNK_SIZE = 500      # characters if using langchain; we also accept sentence based fallback
CHUNK_OVERLAP = 50

# =========================
# STREAMLIT INIT
# =========================
st.set_page_config(page_title="📊 Financial Report RAG Bot", page_icon="📉", layout="wide")
os.makedirs("financial_index", exist_ok=True)

# =========================
# HELPERS: persistence
# =========================
def save_object(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_object(path: str, default: Any):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default

# =========================
# CACHED RESOURCES
# =========================
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model

@st.cache_resource
def load_or_create_faiss_index(dim: int):
    if os.path.exists(INDEX_PATH):
        try:
            idx = faiss.read_index(INDEX_PATH)
            # If dimension mismatch, create new
            if idx.d != dim:
                return faiss.IndexFlatL2(dim)
            return idx
        except Exception:
            return faiss.IndexFlatL2(dim)
    else:
        return faiss.IndexFlatL2(dim)

def load_metadata() -> List[Dict[str, Any]]:
    return load_object(METADATA_PATH, [])

def load_embedding_cache() -> Dict[str, Any]:
    return load_object(EMBEDDING_CACHE_PATH, {})

# =========================
# Initialize model, index, caches
# =========================
model = load_embedding_model()
EMBEDDING_DIM = model.get_sentence_embedding_dimension()
index = load_or_create_faiss_index(EMBEDDING_DIM)
metadata: List[Dict[str, Any]] = load_metadata()
embedding_cache: Dict[str, Any] = load_embedding_cache()

# =========================
# Configure Gemini (Google)
# Put your GEMINI_API_KEY in Streamlit secrets under key 'GEMINI_API_KEY'
# =========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_clients = {name: genai.GenerativeModel(name) for name in GEMINI_MODELS}
except Exception as e:
    gemini_clients = {}
    st.warning("Warning: Gemini API not configured or failed to initialize. Put GEMINI_API_KEY in Streamlit secrets.")

# =========================
# Text cleaning & formatting
# =========================
def clean_text_for_chunks(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def split_text_into_chunks(text: str, chunk_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Prefer langchain RecursiveCharacterTextSplitter if available for robust chunking;
    otherwise fallback to sentence-based splitter that groups sentences into approx. chunk_chars.
    """
    text = clean_text_for_chunks(text)
    if _HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""])
        return splitter.split_text(text)
    # fallback: naive sentence splitter grouped to approximate char length
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_chars:
            current = (current + " " + s).strip()
        else:
            if len(current) > 30:
                chunks.append(current)
            current = s
    if current and len(current) > 30:
        chunks.append(current)
    return chunks

def highlight_numbers(text: str) -> str:
    """Auto-bold numbers, dollar amounts, percentages, and simple decimals (EPS) for Markdown."""
    if not text:
        return text
    # Bold percentages (e.g., 5% or 5.0%)
    text = re.sub(r'(\d+(?:\.\d+)?)\s?%', r'**\1%**', text)
    # Bold dollar amounts like $12,345.67
    text = re.sub(r'\$\s*([\d,]+(?:\.\d+)?)', r'**$\1**', text)
    # Bold plain numbers with commas/decimals (avoid making every small year bold accidentally is acceptable here)
    text = re.sub(r'(?<!\$\*)\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)\b', r'**\1**', text)
    # Avoid wrapping Markdown syntax accidentally: collapse multiple bold markers
    text = re.sub(r'\*\*\s+\*\*', '**', text)
    # Collapse excessive spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_and_normalize_answer(text: str) -> str:
    """General fixes: ensure bullets, fix broken decimals like '12. 5' -> '12.5', ensure spaces around bullets."""
    if not text:
        return text
    # Ensure each '*' bullet starts on a newline
    text = re.sub(r'(?<!\n)\s*\*\s*', r'\n* ', text)
    # Fix broken decimal splits: '12. 5' -> '12.5'
    text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:?!)])', r'\1', text)
    text = re.sub(r'([(!])\s+', r'\1', text)
    # Collapse many spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# Embeddings, PDF processing & index updates
# =========================
def file_hash(content_bytes: bytes) -> str:
    return hashlib.md5(content_bytes).hexdigest()

def get_or_compute_embeddings(chunks: List[str], file_hash_key: str) -> np.ndarray:
    """Return cached embeddings for file hash or compute and store them."""
    global embedding_cache
    if file_hash_key in embedding_cache:
        arr = np.array(embedding_cache[file_hash_key], dtype="float32")
        return arr
    # compute
    vectors = model.encode(chunks, convert_to_numpy=True)
    embedding_cache[file_hash_key] = vectors.tolist()
    try:
        save_object(embedding_cache, EMBEDDING_CACHE_PATH)
    except Exception:
        pass
    return vectors

def process_pdf_bytes(content: bytes, company: str = "unknown") -> Dict[str, Any]:
    """Extract text from PDF, chunk, embed and add to FAISS + metadata."""
    global index, metadata
    fh = file_hash(content)
    # avoid duplicate file ingestion
    if any(m.get("file_hash") == fh for m in metadata):
        return {"success": False, "error": "File already ingested (duplicate)."}

    try:
        raw_text = extract_text(BytesIO(content))
    except Exception as e:
        return {"success": False, "error": f"PDF extraction failed: {e}"}
    if not raw_text or len(raw_text.strip()) < 100:
        return {"success": False, "error": "No text extracted or file too short."}

    clean_text = clean_text_for_chunks(raw_text)
    chunks = split_text_into_chunks(clean_text, chunk_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        return {"success": False, "error": "No chunks produced."}

    vectors = get_or_compute_embeddings(chunks, fh)
    # Add vectors and metadata in same order
    for i, chunk in enumerate(chunks):
        vec = np.array([vectors[i]]).astype("float32")
        index.add(vec)
        metadata.append({
            "id": len(metadata),
            "content": chunk,
            "company": company,
            "file_hash": fh,
            "timestamp": datetime.now().isoformat()
        })
    # persist
    try:
        faiss.write_index(index, INDEX_PATH)
    except Exception:
        pass
    try:
        save_object(metadata, METADATA_PATH)
    except Exception:
        pass

    return {"success": True, "chunks_added": len(chunks)}

# =========================
# Retrieval
# =========================
def retrieve(query: str, top_k: int = 5, companies: Optional[List[str]] = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
    """
    Search FAISS for query nearest neighbors, then filter by companies if supplied.
    Returns up to top_k metadata entries with similarity score (1/(1+dist)).
    """
    if index.ntotal == 0:
        return []

    q_vec = model.encode([query]).astype("float32")
    # search more to allow company filtering
    search_k = min(max(top_k * 3, top_k), index.ntotal)
    D, I = index.search(q_vec, search_k)  # D: distances, I: indices

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        if companies and len(companies) > 0 and meta.get("company") not in companies:
            continue
        similarity = 1.0 / (1.0 + float(dist))
        if similarity < min_score:
            continue
        mcopy = meta.copy()
        mcopy["score"] = similarity
        results.append(mcopy)
        if len(results) >= top_k:
            break
    return results

# =========================
# Gemini call (fallback)
# =========================
def generate_with_gemini(prompt: str, max_tokens: int = 500) -> Tuple[str, str]:
    """Try models in GEMINI_MODELS order, return (text, model_name)."""
    if not gemini_clients:
        return "❌ Gemini not configured.", "none"
    last_err = None
    for name in GEMINI_MODELS:
        client = gemini_clients.get(name)
        if client is None:
            continue
        try:
            response = client.generate_content(prompt, generation_config={"max_output_tokens": max_tokens, "temperature": 0.3})
            # response.text is used by SDK
            text = getattr(response, "text", None)
            if text is None:
                # some SDKs return different shapes; try to str(response) fallback
                text = str(response)
            return text, name
        except Exception as e:
            last_err = e
            # if quota, try next; else break and return error
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            return f"❌ Gemini error: {e}", name
    return f"❌ All Gemini models failed. Last error: {last_err}", "none"

# =========================
# Export helpers
# =========================
def build_history_df(history: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in history:
        q = rec.get("question")
        a = rec.get("answer")
        sources = ", ".join(sorted({c.get("company", "Unknown") for c in rec.get("chunks", [])}))
        ts = rec.get("timestamp")
        rows.append({"Question": q, "Answer": a, "Sources": sources, "Timestamp": ts})
    return pd.DataFrame(rows)

def get_download_bytes_for_df(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="QnA")
        return buffer.getvalue()
    else:
        raise ValueError("Unsupported format")

# =========================
# Session state init
# =========================
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# =========================
# Sidebar: uploads, controls, export
# =========================
st.sidebar.title("⚙️ Document Management")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

company_tag = st.sidebar.text_input("Company tag (optional)", placeholder="e.g., Coca-Cola Q2 2025")

if uploaded_files:
    for f in uploaded_files:
        content = f.read()
        with st.spinner(f"Processing {f.name}..."):
            res = process_pdf_bytes(content, company=company_tag or f.name)
            if res.get("success"):
                st.sidebar.success(f"Added {res.get('chunks_added')} chunks from {f.name}")
            else:
                st.sidebar.error(f"Failed to process {f.name}: {res.get('error')}")
    # After processing, re-persist index & metadata already done inside function, refresh ui
    st.experimental_rerun()

if st.sidebar.button("Reset Index & Data"):
    # remove files, reset in-memory
    for p in (INDEX_PATH, METADATA_PATH, EMBEDDING_CACHE_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    # recreate empty index and metadata
    index.reset()
    metadata.clear()
    embedding_cache.clear()
    st.session_state.history.clear()
    st.sidebar.success("Index, metadata and cache cleared.")
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Processed documents")
if metadata:
    counts: Dict[str, int] = {}
    for m in metadata:
        counts[m.get("company", "unknown")] = counts.get(m.get("company", "unknown"), 0) + 1
    st.sidebar.json(counts)
else:
    st.sidebar.info("No documents processed yet.")

# export buttons (history)
st.sidebar.markdown("---")
st.sidebar.subheader("Export Q&A History")
if st.session_state.history:
    df_hist = build_history_df(st.session_state.history)
    csv_bytes = get_download_bytes_for_df(df_hist, "csv")
    excel_bytes = get_download_bytes_for_df(df_hist, "excel")
    st.sidebar.download_button("Download CSV", csv_bytes, file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
    st.sidebar.download_button("Download Excel", excel_bytes, file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.sidebar.info("No Q&A history to export")

# =========================
# Main UI
# =========================
st.title("📊 Financial Report RAG Bot")
st.markdown(
    "Upload financial PDFs, then ask a question. Answers are retrieved using FAISS + Sentence Transformers and generated with Gemini (with fallback)."
)

# company filter
all_companies = sorted(list({m.get("company", "unknown") for m in metadata}))
selected_companies = st.multiselect("Filter by company (optional)", options=all_companies, default=all_companies if all_companies else [])

query = st.text_input("💬 Ask your financial question:", placeholder="e.g. What was the revenue growth in Q4?")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Context chunks to retrieve", 1, 10, 5)
with col2:
    min_sim = st.slider("Minimum similarity (0-1)", 0.0, 1.0, 0.0, 0.05)

if st.button("Get Answer"):
    if not query or not query.strip():
        st.warning("Please type a question.")
    elif index.ntotal == 0:
        st.warning("Please upload and process at least one PDF first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            retrieved = retrieve(query, top_k=top_k, companies=selected_companies, min_score=min_sim)
            if not retrieved:
                st.info("No relevant information found in the uploaded documents.")
                st.session_state.history.append({"question": query, "answer": "No relevant information found.", "chunks": [], "timestamp": datetime.now().isoformat(), "model": "none"})
            else:
                # Build context with short previews
                context_parts = []
                for r in retrieved:
                    preview = r.get("content", "")[:600].strip()
                    context_parts.append(f"--- Source: {r.get('company','unknown')} (score {r.get('score',0):.3f}) ---\n{preview}")
                context_text = "\n\n".join(context_parts)

                prompt = f"""You are a professional financial analyst assistant. Use the provided context to answer the question clearly and concisely.
Highlight numeric metrics. Cite sources in parentheses like (Source: Company Name).

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

                raw_answer, model_used = generate_with_gemini(prompt, max_tokens=600)
                if not raw_answer:
                    raw_answer = "⚠️ Empty response from Gemini."

                cleaned = clean_and_normalize_answer(raw_answer)
                highlighted = highlight_numbers(cleaned)

                # show the answer
                st.markdown("### 📋 Answer")
                st.markdown(highlighted)

                # Save to history with chunks for traceability
                st.session_state.history.append({
                    "question": query,
                    "answer": highlighted,
                    "chunks": retrieved,
                    "timestamp": datetime.now().isoformat(),
                    "model": model_used
                })

# Show chat / history
st.markdown("---")
st.markdown("## 🗣️ Chat History")
if st.session_state.history:
    # show newest first
    for rec in reversed(st.session_state.history):
        st.markdown(f"**You:** {rec.get('question')}")
        st.markdown(f"**Bot:** {rec.get('answer')}")
        if rec.get("chunks"):
            with st.expander("📚 Sources used (preview)"):
                for c in rec.get("chunks"):
                    st.markdown(f"- **{c.get('company','unknown')}** (score {c.get('score',0):.3f}) → {c.get('content','')[:240]}...")
        st.caption(f"Model: {rec.get('model','unknown')}  •  {rec.get('timestamp')}")
        st.markdown("---")
else:
    st.info("No Q&A yet. Ask a question to get started.")