import streamlit as st
import numpy as np
import faiss
import pickle
import os
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re
from datetime import datetime
import hashlib
import json
import base64
from typing import Tuple

# ==== CONFIG ====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]  # Fallback hierarchy
CHUNK_SIZE = 5
OVERLAP = 1

# ==== INIT ====
st.set_page_config(
    page_title="📊 Financial Report RAG Bot", 
    page_icon="📉", 
    layout="wide",
    initial_sidebar_state="expanded"
)
os.makedirs("financial_index", exist_ok=True)

# Initialize components with caching
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def initialize_faiss_index():
    try:
        if os.path.exists(INDEX_PATH):
            return faiss.read_index(INDEX_PATH)
        else:
            return faiss.IndexFlatL2(384)
    except Exception as e:
        st.error(f"Failed to initialize FAISS index: {e}")
        return faiss.IndexFlatL2(384)

@st.cache_data
def load_embedding_cache():
    try:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}
    except Exception:
        return {}

def save_embedding_cache(cache):
    try:
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        st.warning(f"Could not save embedding cache: {e}")

def load_metadata():
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                return pickle.load(f)
        return []
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return []

# Initialize components
model = load_embedding_model()
index = initialize_faiss_index()
metadata = load_metadata()
embedding_cache = load_embedding_cache()

# Gemini API setup with fallback
gemini_clients = {}
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    for model_name in GEMINI_MODELS:
        gemini_clients[model_name] = genai.GenerativeModel(model_name)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")

# ==== ENHANCED UTILS ====
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    return text.strip()

def clean_gemini_output(text: str) -> str:
    """Clean and auto-highlight all key financial metrics."""
    if not text:
        return "⚠️ No response generated."

    # Ensure bullets are on new lines
    text = re.sub(r'(?<!\n)\* ', r'\n* ', text)

    # Normalize bold formatting
    text = re.sub(r'\*\*\s+', '**', text)
    text = re.sub(r'\s+\*\*', '**', text)

    # Bold percentages
    text = re.sub(r'(\d+(?:\.\d+)?)\s?%', r'**\1%**', text)

    # Bold dollar amounts
    text = re.sub(r'\$([\d,.]+)', r'**$\1**', text)

    # Bold million/billion values
    text = re.sub(r'(\d+(?:\.\d+)?)\s?(million|billion)', r'**\1 \2**', text, flags=re.I)

    # Bold EPS-style decimals
    text = re.sub(r'(?<!\d)(\d+\.\d{1,2})(?!\d)', r'**\1**', text)

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text)

    # Ensure spacing after punctuation
    text = re.sub(r'([.!?])\s+(\*)', r'\1\n\2', text)

    return text.strip()

def create_smart_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk_sentences = sentences[i:i + chunk_size]
        if chunk_sentences:
            chunk = ' '.join(chunk_sentences)
            if len(chunk) > 50:
                chunks.append(chunk)
    return chunks

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

@st.cache_data
def get_cached_embeddings(chunks: tuple, _file_hash: str) -> np.ndarray:
    global embedding_cache
    if _file_hash in embedding_cache:
        return np.array(embedding_cache[_file_hash])
    if model:
        vectors = model.encode(list(chunks))
        embedding_cache[_file_hash] = vectors.tolist()
        save_embedding_cache(embedding_cache)
        return vectors
    return None

def retrieve(query: str, top_k: int = 5, min_score: float = 0.3) -> list:
    if not model or index.ntotal == 0:
        return []
    try:
        q_vec = model.encode([query])
        D, I = index.search(np.array(q_vec).astype("float32"), min(top_k, index.ntotal))
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(metadata):
                similarity = 1 / (1 + distance)
                if similarity >= min_score:
                    chunk_data = metadata[idx].copy()
                    chunk_data['similarity'] = similarity
                    results.append(chunk_data)
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def generate_with_gemini_fallback(prompt: str, max_tokens: int = 500) -> Tuple[str, str]:
    if not gemini_clients:
        return "❌ Gemini API not configured properly.", "none"
    for model_name in GEMINI_MODELS:
        try:
            client = gemini_clients[model_name]
            response = client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            )
            return response.text if response.text else "⚠️ Empty response from Gemini.", model_name
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            else:
                return f"❌ Gemini error: {str(e)}", model_name
    return "❌ All Gemini models failed or quota exceeded.", "none"

def add_source_references(answer: str, chunks: list) -> str:
    if not chunks:
        return answer
    source_text = "\n\n**📚 Sources:**\n"
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get('company', 'Unknown')
        similarity = chunk.get('similarity', 0)
        source_text += f"- **[Source {i}]** {company} (Relevance: {similarity:.1%})\n"
    return answer + source_text

def process_pdf(file_content: bytes, company: str = "unknown") -> dict:
    if not model:
        return {"success": False, "error": "Embedding model not loaded"}
    try:
        file_hash = get_file_hash(file_content)
        if any(m.get('file_hash') == file_hash for m in metadata):
            return {"success": False, "error": "This file has already been processed"}
        text = extract_text(BytesIO(file_content))
        if not text or len(text.strip()) < 100:
            return {"success": False, "error": "Could not extract sufficient text from PDF"}
        text = clean_text(text)
        chunks = create_smart_chunks(text, CHUNK_SIZE, OVERLAP)
        if not chunks:
            return {"success": False, "error": "No valid chunks created from text"}
        vectors = get_cached_embeddings(tuple(chunks), file_hash)
        if vectors is None:
            return {"success": False, "error": "Failed to generate embeddings"}
        added_count = 0
        for i, chunk in enumerate(chunks):
            try:
                index.add(np.array([vectors[i]]).astype("float32"))
                metadata.append({
                    "id": len(metadata),
                    "content": chunk,
                    "company": company,
                    "file_hash": file_hash,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_index": i
                })
                added_count += 1
            except Exception as e:
                st.warning(f"Failed to add chunk {i}: {e}")
        faiss.write_index(index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        return {
            "success": True, 
            "chunks_added": added_count,
            "total_text_length": len(text)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==== SESSION STATE ====
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ==== UI ====
st.title("📊 Financial Report RAG Bot")
st.markdown("Upload financial PDFs and ask questions. All key financial metrics will be **auto-highlighted** in bold.")

query = st.text_input("💬 Ask your financial question:")

if query and len(metadata) > 0:
    with st.spinner("🔍 Searching and generating answer..."):
        chunks = retrieve(query)
        if chunks:
            context = "\n\n".join([f"[{c['company']}] {c['content']}" for c in chunks])
            prompt = f"""You are a financial analyst assistant. Use the context to answer clearly with metrics bolded.

CONTEXT:
{context}

QUESTION: {query}"""
            answer, model_used = generate_with_gemini_fallback(prompt, max_tokens=600)
            answer = clean_gemini_output(answer)
            answer = add_source_references(answer, chunks)
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    if query:
        st.info("📤 Please upload some financial documents first.")