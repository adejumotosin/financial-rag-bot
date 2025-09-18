import streamlit as st
import numpy as np
import faiss
import pickle
import os
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import re
import hashlib
import json
import logging
from datetime import datetime
import google.generativeai as genai
from typing import List, Dict, Any, Optional

# ==== CONFIG ====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
HASHES_PATH = "financial_index/content_hashes.pkl"

# API key setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== INIT STREAMLIT ====
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
                logger.warning("Index dimension mismatch, creating new index.")
                return faiss.IndexFlatL2(dim)
            return index
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return faiss.IndexFlatL2(dim)
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
        logger.error(f"Error loading hashes: {e}")
    return set()

def save_content_hashes(hashes: set):
    try:
        with open(HASHES_PATH, "wb") as f:
            pickle.dump(hashes, f)
    except Exception as e:
        logger.error(f"Error saving hashes: {e}")

# Load state
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
    chunks, current_chunk = [], ""
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
        logger.error(f"Retrieval error: {e}")
        return []

# ==== GEMINI GENERATION ====
def generate_with_gemini(prompt: str, max_tokens: int = 500) -> str:
    """Generate text with Gemini, clean formatting, bullet points for financial metrics."""
    def clean_and_format(text: str) -> str:
        cleaned = (
            text.replace("\n", " ")
            .replace("  ", " ")
            .replace(" ,", ",")
            .replace(" .", ".")
            .strip()
        )
        sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
        if any(word in cleaned.lower() for word in ["revenue", "income", "growth", "margin", "eps", "%"]):
            formatted = "\n".join([f"- {s}." for s in sentences])
            return formatted
        return cleaned

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": max_tokens,
            },
        )
        return clean_and_format(response.text)
    except Exception:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": max_tokens,
                },
            )
            return clean_and_format(response.text)
        except Exception as e2:
            return f"❌ Gemini fallback failed: {e2}"

# ==== PDF PROCESS ====
def process_pdf(file, company: str = "unknown") -> Dict[str, int]:
    try:
        text = extract_text(file)
        text = clean_text(text)
        if len(text.strip()) < 100:
            return {"added": 0, "duplicates": 0, "error": "PDF appears empty"}
        chunks = create_smart_chunks(text)
        new_chunks, duplicate_count = [], 0
        for chunk in chunks:
            chunk_hash = get_content_hash(chunk)
            if chunk_hash not in content_hashes:
                new_chunks.append(chunk)
                content_hashes.add(chunk_hash)
            else:
                duplicate_count += 1
        batch_size, added_count = 32, 0
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

# ==== STREAMLIT UI ====
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("⚙️ Financial Docs Management")
uploaded = st.sidebar.file_uploader("Upload Financial PDF(s)", type="pdf", accept_multiple_files=True)
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
    st.sidebar.success(f"📊 Total: {total_added} new, {total_duplicates} duplicates")

st.sidebar.subheader("🗄️ Index Management")
st.sidebar.metric("Total Chunks", len(metadata))
st.sidebar.metric("Index Size", f"{index.ntotal if hasattr(index, 'ntotal') else 0}")
companies = list(set([item.get('company', 'unknown') for item in metadata]))
if companies:
    st.sidebar.write(f"🏢 Companies: {', '.join(sorted(companies))}")
if st.sidebar.button("🧹 Reset Index"):
    try:
        for path in [INDEX_PATH, METADATA_PATH, HASHES_PATH]:
            if os.path.exists(path): os.remove(path)
        index.reset(); metadata.clear(); content_hashes.clear(); st.session_state.history.clear()
        st.sidebar.success("✅ Index reset!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Error resetting: {e}")

# ==== MAIN ====
st.title("📊 Financial Report RAG Bot")
st.markdown("Upload financial PDFs and ask questions. Gemini will summarize with context.")

col1, col2, col3 = st.columns(3)
with col1: st.metric("📄 Documents", len(set(item.get('company','unknown') for item in metadata)))
with col2: st.metric("🧩 Chunks", len(metadata))
with col3: st.metric("💬 Questions Asked", len(st.session_state.history))

if len(metadata) == 0:
    st.warning("👈 Upload some PDF documents to get started!")
else:
    st.success(f"✅ Ready with {len(metadata)} chunks from {len(companies)} companies!")

query = st.text_input("💬 Ask your question:", placeholder="e.g., What was the revenue growth in Q4?")

if query and len(metadata) > 0:
    with st.spinner("🔍 Searching and generating answer..."):
        chunks = retrieve(query, top_k=5)
        if not chunks:
            st.warning("No relevant information found.")
        else:
            context = "\n\n".join([f"[{chunk['company']}]: {chunk['content']}" for chunk in chunks])
            prompt = f"""You are a financial analyst assistant. Use context from documents to answer.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, accurate, professional answer
- Include financial metrics and percentages
- If incomplete, note missing info
- Summarize cleanly

ANSWER:"""
            answer = generate_with_gemini(prompt, max_tokens=400)
            st.markdown(f"**Answer:**\n\n{answer}")
            st.session_state.history.append((query, answer, chunks))

if st.session_state.history:
    with st.expander("📄 Sources Used"):
        for i, chunk in enumerate(st.session_state.history[-1][2], 1):
            relevance = "High" if chunk.get('score', 1) < 0.5 else "Medium" if chunk.get('score', 1) < 1.0 else "Low"
            st.markdown(f"**Source {i}** - *{chunk.get('company','N/A')}* | Relevance: {relevance}")
            st.caption(chunk['content'][:300] + ("..." if len(chunk['content'])>300 else ""))