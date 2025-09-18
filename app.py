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
from typing import Dict, List, Tuple, Optional

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
    """Load sentence transformer model with caching."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def initialize_faiss_index():
    """Initialize FAISS index with caching."""
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
    """Load embedding cache for faster processing."""
    try:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        return {}

def save_embedding_cache(cache):
    """Save embedding cache to disk."""
    try:
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        st.warning(f"Could not save embedding cache: {e}")

def load_metadata():
    """Load metadata with error handling."""
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
    """Clean extracted text for better processing."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    return text.strip()

def clean_gemini_output(text: str) -> str:
    """Clean Gemini output for proper Markdown rendering."""
    if not text:
        return "⚠️ No response generated."
    
    text = re.sub(r'\*\*\s+', '**', text)
    text = re.sub(r'\s+\*\*', '**', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text.strip()

def create_smart_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """Create overlapping chunks with sentence boundaries."""
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
    """Generate hash for file content to detect duplicates."""
    return hashlib.md5(file_content).hexdigest()

@st.cache_data
def get_cached_embeddings(chunks: tuple, _file_hash: str) -> np.ndarray:
    """Get cached embeddings or compute new ones."""
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
    """Retrieve top_k chunks from FAISS index with similarity filtering."""
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
    """Call Gemini with fallback models and return (response, model_used)."""
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
                continue  # Try next model
            else:
                return f"❌ Gemini error: {str(e)}", model_name
    
    return "❌ All Gemini models failed or quota exceeded.", "none"

def add_source_references(answer: str, chunks: list) -> str:
    """Add source references to the answer."""
    # Simple approach: add source references at the end
    if not chunks:
        return answer
    
    source_text = "\n\n**📚 Sources:**\n"
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get('company', 'Unknown')
        similarity = chunk.get('similarity', 0)
        source_text += f"- **[Source {i}]** {company} (Relevance: {similarity:.1%})\n"
    
    return answer + source_text

def process_pdf(file_content: bytes, company: str = "unknown") -> dict:
    """Extract text, split into chunks, embed & add to FAISS index."""
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
        
        # Use cached embeddings
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

def export_chat_history(format_type: str = "markdown") -> str:
    """Export chat history in different formats."""
    if not st.session_state.history:
        return "No chat history to export."
    
    if format_type == "markdown":
        content = "# Financial Report RAG Bot - Chat History\n\n"
        content += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, entry in enumerate(st.session_state.history, 1):
            content += f"## Question {i}\n**Q:** {entry['question']}\n\n"
            content += f"**A:** {entry['answer']}\n\n"
            
            if 'chunks' in entry and entry['chunks']:
                content += "**Sources:**\n"
                for j, chunk in enumerate(entry['chunks'], 1):
                    content += f"{j}. {chunk.get('company', 'Unknown')}: {chunk['content'][:100]}...\n"
            content += "\n---\n\n"
        
        return content
    
    elif format_type == "json":
        return json.dumps(st.session_state.history, indent=2, default=str)
    
    return "Unsupported format"

def get_download_link(content: str, filename: str, format_type: str = "text") -> str:
    """Generate download link for content."""
    if format_type == "text":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "text/plain"
    elif format_type == "json":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "application/json"
    else:
        b64 = base64.b64encode(content.encode()).decode()
        mime = "text/plain"
    
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# ==== SESSION STATE ====
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ==== SIDEBAR ====
st.sidebar.title("⚙️ Document Management")

# File upload section
with st.sidebar.expander("📂 Upload Financial Documents", expanded=True):
    uploaded = st.file_uploader(
        "Select PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload financial reports, earnings statements, or other financial documents"
    )
    
    company_tag = st.text_input(
        "🏷️ Company Name", 
        placeholder="e.g., Apple Inc.",
        help="Tag documents with company name for better organization"
    )
    
    if uploaded:
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded):
            file_key = f"{file.name}_{company_tag or 'unknown'}"
            if file_key not in st.session_state.processed_files:
                with st.spinner(f"🔄 Processing {file.name}..."):
                    result = process_pdf(file.read(), company=company_tag or file.name.split('.')[0])
                    
                    if result["success"]:
                        st.success(f"✅ {file.name}: Added {result['chunks_added']} chunks")
                        st.session_state.processed_files.add(file_key)
                    else:
                        st.error(f"❌ {file.name}: {result['error']}")
            
            progress_bar.progress((i + 1) / len(uploaded))

# Export section
with st.sidebar.expander("📤 Export & Download"):
    if st.session_state.history:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download MD"):
                content = export_chat_history("markdown")
                filename = f"financial_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                st.markdown(get_download_link(content, filename, "text"), unsafe_allow_html=True)
        
        with col2:
            if st.button("📊 Download JSON"):
                content = export_chat_history("json")
                filename = f"financial_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                st.markdown(get_download_link(content, filename, "json"), unsafe_allow_html=True)
    else:
        st.info("No chat history to export")

# Index management
with st.sidebar.expander("🗃️ Index Management"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Clear Index", type="secondary"):
            try:
                files_to_remove = [INDEX_PATH, METADATA_PATH, EMBEDDING_CACHE_PATH]
                for file_path in files_to_remove:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                index.reset()
                metadata.clear()
                st.session_state.history.clear()
                st.session_state.processed_files.clear()
                
                # Clear caches
                st.cache_data.clear()
                
                st.success("Index cleared successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing index: {e}")
    
    with col2:
        if st.button("📊 Show Stats"):
            companies = list(set(m.get('company', 'Unknown') for m in metadata))
            st.info(f"""
            **Index Statistics:**
            - Total chunks: {len(metadata)}
            - Companies: {len(companies)}
            - Files processed: {len(st.session_state.processed_files)}
            - Cache size: {len(embedding_cache)} files
            """)

# ==== MAIN INTERFACE ====
st.title("📊 Financial Report RAG Bot")
st.markdown("""
Upload financial PDFs and ask questions about the content. The bot uses **Retrieval-Augmented Generation (RAG)** 
to provide accurate answers based on your documents.
""")

# Display statistics
if metadata:
    companies = list(set(m.get('company', 'Unknown') for m in metadata))
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Documents", len(st.session_state.processed_files))
    with col2:
        st.metric("🧩 Text Chunks", len(metadata))
    with col3:
        st.metric("🏢 Companies", len(companies))
    with col4:
        st.metric("💬 Questions Asked", len(st.session_state.history))
    
    with st.expander("📋 Companies in Index"):
        st.write(", ".join(companies))

# Query interface
query = st.text_input(
    "💬 Ask your financial question:",
    placeholder="e.g., What were the Q3 revenue figures? How did expenses change year-over-year?",
    help="Ask specific questions about financial metrics, trends, or company performance"
)

# Search parameters
with st.expander("🔧 Advanced Search Settings"):
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of context chunks", 1, 10, 5)
    with col2:
        min_similarity = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.3, 0.1)

# Process query
if query and len(metadata) > 0:
    with st.spinner("🔍 Searching documents and generating answer..."):
        chunks = retrieve(query, top_k=top_k, min_score=min_similarity)
        
        if not chunks:
            st.warning("⚠️ No relevant information found. Try rephrasing your question or adjusting the similarity threshold.")
        else:
            context = "\n\n".join([
                f"[{c['company']}] (Similarity: {c['similarity']:.2f}): {c['content']}" 
                for c in chunks
            ])
            
            prompt = f"""You are a professional financial analyst assistant. Analyze the provided context from financial documents to answer the user's question accurately and comprehensively.

CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, well-structured answer using bullet points where appropriate
- **Bold** key financial metrics, numbers, and important findings
- Include specific data points and percentages when available
- If comparing periods, clearly state the timeframes
- Cite which company/document the information comes from when relevant
- If information is incomplete or missing, explicitly state this
- Use professional financial terminology appropriately
- Keep the response focused and avoid speculation beyond the provided data

ANSWER:"""

            answer, model_used = generate_with_gemini_fallback(prompt, max_tokens=600)
            answer = clean_gemini_output(answer)
            answer = add_source_references(answer, chunks)
            
            # Use Streamlit chat interface
            with st.chat_message("assistant"):
                st.markdown(answer)
                if model_used != "none":
                    st.caption(f"Generated using {model_used}")
            
            # Save to history
            st.session_state.history.append({
                "question": query,
                "answer": answer,
                "chunks": chunks,
                "model_used": model_used,
                "timestamp": datetime.now().isoformat()
            })
            
            # Show detailed sources
            with st.expander("📄 Detailed Source Documents", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"""
                    **Source {i}** - *{chunk.get('company', 'Unknown Company')}* 
                    (Relevance: {chunk.get('similarity', 0):.1%})
                    
                    {chunk['content']}
                    """)
                    st.markdown("---")

elif query and len(metadata) == 0:
    st.info("📤 Please upload some financial documents first to start asking questions!")

# ==== ENHANCED CHAT HISTORY ====
if st.session_state.history:
    st.markdown("## 🗣️ Chat History")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🧹 Clear Chat History", type="secondary"):
            st.session_state.history.clear()
            st.experimental_rerun()
    
    # Display using modern chat interface
    for i, entry in enumerate(reversed(st.session_state.history)):
        # User message
        with st.chat_message("user"):
            st.markdown(entry['question'])
        
        # Assistant response
        with st.chat_message("assistant"):
            st.markdown(entry['answer'])
            
            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"🤖 Model: {entry.get('model_used', 'unknown')}")
            with col2:
                st.caption(f"📄 Sources: {len(entry.get('chunks', []))}")
            with col3:
                timestamp = entry.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.caption(f"🕒 {dt.strftime('%H:%M:%S')}")

# ==== FOOTER ====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
🚀 Enhanced Financial RAG Bot | Powered by Sentence Transformers, FAISS & Google Gemini | Built with Streamlit<br>
✨ Features: Smart Caching • Model Fallback • Export Tools • Modern UI
</div>
""", unsafe_allow_html=True)