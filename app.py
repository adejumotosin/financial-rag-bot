import streamlit as st
import numpy as np
import faiss
import pickle
import os
import re
import hashlib
import json
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import google.generativeai as genai

# ==== CONFIG ====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
HASHES_PATH = "financial_index/content_hashes.pkl"

# Gemini setup
GEMINI_MODEL_PRIMARY = "gemini-1.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-1.5-pro"
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== INIT ====
st.set_page_config(
    page_title="📊 Financial Report RAG Bot",
    page_icon="📉",
    layout="wide"
)

os.makedirs("financial_index", exist_ok=True)

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_index():
    model = load_model()
    dim = model.get_sentence_embedding_dimension()
    if os.path.exists(INDEX_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            if index.d != dim:
                logger.warning("Index dim mismatch, creating new index.")
                return faiss.IndexFlatL2(dim)
            return index
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return faiss.IndexFlatL2(dim)
    else:
        return faiss.IndexFlatL2(dim)

def load_metadata():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
    return []

def load_content_hashes():
    try:
        if os.path.exists(HASHES_PATH):
            with open(HASHES_PATH, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading content hashes: {e}")
    return set()

def save_content_hashes(hashes: set):
    try:
        with open(HASHES_PATH, "wb") as f:
            pickle.dump(hashes, f)
    except Exception as e:
        logger.error(f"Error saving content hashes: {e}")

# Initialize
model = load_model()
index = load_index()
metadata = load_metadata()
content_hashes = load_content_hashes()

# ==== UTILS ====
def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text.strip()

def create_smart_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_sentences = max(1, overlap // 100)
            overlap_text = '. '.join(current_chunk.split('. ')[-overlap_sentences:]) if overlap > 0 else ""
            current_chunk = (overlap_text + ". " if overlap_text else "") + sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [chunk for chunk in chunks if len(chunk.strip()) > 20]

def retrieve(query: str, top_k: int = 5, company_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        if len(metadata) == 0 or index.ntotal == 0:
            return []
        q_vec = model.encode([query])
        search_k = min(top_k * 3, len(metadata), index.ntotal)
        D, I = index.search(np.array(q_vec).astype("float32"), search_k)
        results = []
        for i, distance in zip(I[0], D[0]):
            if 0 <= i < len(metadata):
                chunk = metadata[i].copy()
                chunk['score'] = float(distance)
                if company_filter and company_filter != "All":
                    if company_filter.lower() not in chunk.get('company', '').lower():
                        continue
                results.append(chunk)
        results.sort(key=lambda x: x['score'])
        return results[:top_k]
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return []

def generate_with_gemini(prompt: str, max_tokens: int = 500) -> str:
    try:
        response = genai.GenerativeModel(GEMINI_MODEL_PRIMARY).generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower():
            try:
                response = genai.GenerativeModel(GEMINI_MODEL_FALLBACK).generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_output_tokens": max_tokens,
                    },
                )
                return response.text
            except Exception as e2:
                return f"❌ Gemini fallback (Pro) failed: {e2}"
        return f"❌ Gemini error: {e}"

def process_pdf(file, company: str = "unknown") -> Dict[str, int]:
    try:
        text = extract_text(file)
        text = clean_text(text)
        if len(text.strip()) < 100:
            return {"added": 0, "duplicates": 0, "error": "PDF empty or unreadable."}
        chunks = create_smart_chunks(text)
        new_chunks = []
        duplicate_count = 0
        for chunk in chunks:
            chunk_hash = get_content_hash(chunk)
            if chunk_hash not in content_hashes:
                new_chunks.append(chunk)
                content_hashes.add(chunk_hash)
            else:
                duplicate_count += 1
        if not new_chunks:
            return {"added": 0, "duplicates": duplicate_count, "error": "All chunks were duplicates."}
        batch_size = 32
        added_count = 0
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i + batch_size]
            vectors = model.encode(batch_chunks)
            for j, chunk in enumerate(batch_chunks):
                index.add(np.array([vectors[j]]).astype("float32"))
                metadata.append({
                    "id": len(metadata),
                    "content": chunk,
                    "company": company,
                    "chunk_index": i + j,
                    "timestamp": datetime.now().isoformat()
                })
                added_count += 1
        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        save_content_hashes(content_hashes)
        return {"added": added_count, "duplicates": duplicate_count, "error": None}
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        return {"added": 0, "duplicates": 0, "error": str(e)}

def export_chat_history() -> str:
    if not st.session_state.history:
        return "No chat history available."
    export_text = "Financial RAG Bot - Chat History\n"
    export_text += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += "=" * 50 + "\n\n"
    for i, (query, answer, chunks) in enumerate(st.session_state.history, 1):
        export_text += f"Question {i}: {query}\n"
        export_text += f"Answer {i}: {answer}\n"
        export_text += f"Sources: {len(chunks)} chunks from {', '.join(set(c.get('company', 'N/A') for c in chunks))}\n"
        export_text += "-" * 30 + "\n\n"
    return export_text

# ==== SESSION STATE ====
if "history" not in st.session_state:
    st.session_state.history = []

# ==== SIDEBAR ====
st.sidebar.title("⚙️ Financial Docs Management")

uploaded = st.sidebar.file_uploader(
    "Upload Financial PDF(s)",
    type="pdf",
    accept_multiple_files=True
)
company_tag = st.sidebar.text_input("🏷️ Company Name", placeholder="e.g., Apple Inc.")

if uploaded:
    total_added, total_duplicates = 0, 0
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    for i, file in enumerate(uploaded):
        status_text.text(f"Processing {file.name}...")
        progress_bar.progress((i + 1) / len(uploaded))
        result = process_pdf(BytesIO(file.read()), company=company_tag or file.name.replace('.pdf', ''))
        total_added += result["added"]
        total_duplicates += result["duplicates"]
        if result["error"]:
            st.sidebar.error(f"❌ {file.name}: {result['error']}")
        elif result["added"] > 0:
            st.sidebar.success(f"✅ {file.name}: {result['added']} chunks added")
        if result["duplicates"] > 0:
            st.sidebar.info(f"ℹ️ {file.name}: {result['duplicates']} duplicates skipped")
    status_text.text("Processing complete!")
    st.sidebar.success(f"📊 Total: {total_added} new chunks, {total_duplicates} duplicates skipped")

st.sidebar.subheader("🗄️ Index Management")
st.sidebar.metric("Total Chunks", len(metadata))
st.sidebar.metric("Index Size", f"{index.ntotal if hasattr(index, 'ntotal') else 0}")
companies = list(set([item.get('company', 'unknown') for item in metadata]))
if companies:
    st.sidebar.write(f"🏢 **Companies ({len(companies)}):**")
    st.sidebar.write(", ".join(sorted(companies)))

if st.sidebar.button("🧹 Reset Index", type="secondary"):
    for path in [INDEX_PATH, METADATA_PATH, HASHES_PATH]:
        if os.path.exists(path):
            os.remove(path)
    index.reset()
    metadata.clear()
    content_hashes.clear()
    st.session_state.history.clear()
    st.sidebar.success("✅ Index completely reset!")
    st.rerun()

st.sidebar.subheader("📥 Export Options")
if st.session_state.history:
    chat_export = export_chat_history()
    st.sidebar.download_button(
        label="💾 Download Chat History",
        data=chat_export,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

st.sidebar.subheader("🎛️ Search Options")
company_filter = st.sidebar.selectbox("Filter by Company", options=["All"] + sorted(companies), index=0)
top_k = st.sidebar.slider("Results to retrieve", 1, 10, 5)

# ==== MAIN INTERFACE ====
st.title("📊 Financial Report RAG Bot")
st.markdown("Upload PDF financial reports and ask questions. The bot retrieves chunks and answers with Gemini.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 Documents", len(set(item.get('company', 'unknown') for item in metadata)))
with col2:
    st.metric("🧩 Chunks", len(metadata))
with col3:
    st.metric("💬 Questions Asked", len(st.session_state.history))

if len(metadata) == 0:
    st.warning("👈 Upload some PDF documents to get started!")
else:
    st.success(f"✅ Ready with {len(metadata)} chunks from {len(companies)} companies!")

query = st.text_input("💬 Ask your question:", placeholder="e.g., What was the revenue growth in Q4?")
if query and len(metadata) > 0:
    with st.spinner("🔍 Searching and generating answer..."):
        filter_company = None if company_filter == "All" else company_filter
        chunks = retrieve(query, top_k=top_k, company_filter=filter_company)
        if not chunks:
            st.warning("🔍 No relevant information found.")
        else:
            context = "\n\n".join([f"[{chunk['company']}]: {chunk['content']}" for chunk in chunks])
            prompt = f"""You are a financial analyst assistant. Use the provided context from financial documents to answer the question accurately.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
            answer = generate_with_gemini(prompt, max_tokens=300)
            st.markdown(f"**Answer:** {answer}")
            st.session_state.history.append((query, answer, chunks))

if st.session_state.history:
    latest_query, latest_answer, latest_chunks = st.session_state.history[-1]
    with st.expander("📄 Sources Used", expanded=False):
        for i, chunk in enumerate(latest_chunks, 1):
            relevance = "High" if chunk.get('score', 1) < 0.5 else "Medium" if chunk.get('score', 1) < 1.0 else "Low"
            st.markdown(f"""
            **Source {i}** - *{chunk.get('company', 'N/A')}*  
            **Relevance:** {relevance} (Score: {chunk.get('score', 0):.3f})  
            **Content:** {chunk['content'][:400]}{'...' if len(chunk['content']) > 400 else ''}
            """)
            if i < len(latest_chunks):
                st.markdown("---")

if len(st.session_state.history) > 1:
    with st.expander(f"📚 Previous Questions ({len(st.session_state.history) - 1})", expanded=False):
        for i, (q, a, chunks_used) in enumerate(reversed(st.session_state.history[:-1]), 1):
            companies_used = set(c.get('company', 'N/A') for c in chunks_used)
            st.markdown(f"**Q{len(st.session_state.history) - i}:** {q}")
            st.markdown(f"**A:** {a[:200]}{'...' if len(a) > 200 else ''}")
            st.markdown(f"*Sources: {len(chunks_used)} chunks from {', '.join(companies_used)}*")
            st.markdown("---")

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
<small>
Financial RAG Bot v2.0 | Powered by FAISS, Sentence Transformers & Gemini<br/>
Model: {EMBEDDING_MODEL} | Chunks: {len(metadata)} | Companies: {len(companies)}
</small>
</div>
""", unsafe_allow_html=True)