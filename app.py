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

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]  # Fallback hierarchy
CHUNK_SIZE = 5
OVERLAP = 1

# =========================
# STREAMLIT INIT
# =========================
st.set_page_config(
    page_title="📊 Financial Report RAG Bot",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs("financial_index", exist_ok=True)

# =========================
# CACHED INITIALIZERS
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def initialize_faiss_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        return faiss.IndexFlatL2(384)

@st.cache_data
def load_embedding_cache():
    if os.path.exists(EMBEDDING_CACHE_PATH):
        with open(EMBEDDING_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_embedding_cache(cache):
    with open(EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return []

# =========================
# INITIALIZE COMPONENTS
# =========================
model = load_embedding_model()
index = initialize_faiss_index()
metadata = load_metadata()
embedding_cache = load_embedding_cache()

# Gemini API
gemini_clients = {}
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
for model_name in GEMINI_MODELS:
    gemini_clients[model_name] = genai.GenerativeModel(model_name)

# =========================
# UTILS
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def highlight_numbers(text: str) -> str:
    """Auto-highlight numbers and percentages in bold."""
    if not text:
        return text
    text = re.sub(r'(\d+(\.\d+)?%)', r'**\1**', text)  # percentages
    text = re.sub(r'(\$?\d[\d,\.]*)', r'**\1**', text)  # numbers, $numbers
    return text

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
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
    vectors = model.encode(list(chunks))
    embedding_cache[_file_hash] = vectors.tolist()
    save_embedding_cache(embedding_cache)
    return vectors

def retrieve(query: str, top_k: int = 5) -> list:
    if index.ntotal == 0:
        return []
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]

def generate_with_gemini(prompt: str, max_tokens: int = 500):
    for model_name in GEMINI_MODELS:
        try:
            client = gemini_clients[model_name]
            response = client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.3,
                }
            )
            return response.text, model_name
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            return f"❌ Gemini error: {str(e)}", model_name
    return "❌ All Gemini models failed.", "none"

def process_pdf(file_content: bytes, company: str = "unknown"):
    file_hash = get_file_hash(file_content)
    if any(m.get("file_hash") == file_hash for m in metadata):
        return {"success": False, "error": "File already processed"}
    text = extract_text(BytesIO(file_content))
    if not text or len(text.strip()) < 100:
        return {"success": False, "error": "No text extracted"}
    text = clean_text(text)
    chunks = create_chunks(text)
    vectors = get_cached_embeddings(tuple(chunks), file_hash)
    for i, chunk in enumerate(chunks):
        index.add(np.array([vectors[i]]).astype("float32"))
        metadata.append({
            "id": len(metadata),
            "content": chunk,
            "company": company,
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat(),
        })
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    return {"success": True, "chunks": len(chunks)}

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Document Management")
uploaded = st.sidebar.file_uploader("📂 Upload Financial PDF", type="pdf")
company_tag = st.sidebar.text_input("🏷️ Company Name")

if uploaded:
    with st.spinner("🔄 Processing PDF..."):
        result = process_pdf(uploaded.read(), company=company_tag or "unknown")
        if result["success"]:
            st.sidebar.success(f"✅ Added {result['chunks']} chunks")
        else:
            st.sidebar.error(f"❌ {result['error']}")

if st.sidebar.button("🧹 Reset Index"):
    if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
    if os.path.exists(EMBEDDING_CACHE_PATH): os.remove(EMBEDDING_CACHE_PATH)
    index.reset()
    metadata.clear()
    embedding_cache.clear()
    st.session_state.history.clear()
    st.sidebar.success("Index reset")

# =========================
# MAIN
# =========================
st.title("📊 Financial Report RAG Bot")
st.markdown("Upload financial PDFs and ask questions. Gemini will summarize with context.")

query = st.text_input("💬 Ask your financial question:")

if query:
    with st.spinner("Thinking..."):
        chunks = retrieve(query, top_k=5)
        context = "\n\n".join(chunk["content"] for chunk in chunks)
        prompt = f"""
You are a financial assistant. Use the context to answer.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""
        answer, model_used = generate_with_gemini(prompt)
        answer = highlight_numbers(answer)
        st.session_state.history.append((query, answer))

if st.session_state.history:
    st.markdown("## 🗣️ Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")