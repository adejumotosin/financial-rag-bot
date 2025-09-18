import streamlit as st import numpy as np import faiss import pickle import os import pandas as pd from io import BytesIO from pdfminer.high_level import extract_text from sentence_transformers import SentenceTransformer import google.generativeai as genai import re import json from datetime import datetime import hashlib from langchain.text_splitter import RecursiveCharacterTextSplitter from typing import List, Dict, Any

=========================

CONFIG

=========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" INDEX_PATH = "financial_index/faiss_index.bin" METADATA_PATH = "financial_index/metadata.pkl" EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl" GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"] CHUNK_SIZE = 500 CHUNK_OVERLAP = 50

=========================

STREAMLIT INIT

=========================

st.set_page_config( page_title="📊 Financial Report RAG Bot", page_icon="📉", layout="wide", initial_sidebar_state="expanded" )

os.makedirs("financial_index", exist_ok=True)

=========================

HELPERS

=========================

def clean_text(text: str) -> str: return re.sub(r'\s+', ' ', text).strip()

def highlight_numbers(text: str) -> str: if not text: return text text = re.sub(r'(\d+(.\d+)?%)', r'\1', text) text = re.sub(r'($?\d[\d,.]*)', r'\1', text) return text

def get_file_hash(file_content: bytes) -> str: return hashlib.md5(file_content).hexdigest()

def save_object(obj: Any, path: str): with open(path, "wb") as f: pickle.dump(obj, f)

def load_object(path: str, default: Any): if os.path.exists(path): with open(path, "rb") as f: return pickle.load(f) return default

=========================

CACHE LOADERS

=========================

@st.cache_resource def load_embedding_model(): return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource def load_faiss_index(): return faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else faiss.IndexFlatL2(384)

@st.cache_resource def initialize_gemini_clients(): genai.configure(api_key=st.secrets["GEMINI_API_KEY"]) return {m: genai.GenerativeModel(m) for m in GEMINI_MODELS}

@st.cache_data def load_metadata_cached(): return load_object(METADATA_PATH, [])

@st.cache_data def load_embedding_cache_cached(): return load_object(EMBEDDING_CACHE_PATH, {})

=========================

RAG BOT CLASS

=========================

class FinancialRAGBot: def init(self, model, index, metadata, embedding_cache, gemini_clients): self.model = model self.index = index self.metadata = metadata self.embedding_cache = embedding_cache self.gemini_clients = gemini_clients

def _get_cached_embeddings(self, chunks: tuple, _file_hash: str) -> np.ndarray:
    if _file_hash in self.embedding_cache:
        return np.array(self.embedding_cache[_file_hash])
    vectors = self.model.encode(list(chunks))
    self.embedding_cache[_file_hash] = vectors.tolist()
    save_object(self.embedding_cache, EMBEDDING_CACHE_PATH)
    return vectors

def process_pdf(self, file_content: bytes, company: str = "unknown"):
    file_hash = get_file_hash(file_content)
    if any(m.get("file_hash") == file_hash for m in self.metadata):
        return {"success": False, "error": "File already processed"}
    try:
        text = extract_text(BytesIO(file_content))
        if not text or len(text.strip()) < 100:
            return {"success": False, "error": "No text extracted or text too short"}
    except Exception as e:
        return {"success": False, "error": f"Error extracting text: {str(e)}"}

    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    vectors = self._get_cached_embeddings(tuple(chunks), file_hash)

    for i, chunk in enumerate(chunks):
        self.index.add(np.array([vectors[i]]).astype("float32"))
        self.metadata.append({
            "id": len(self.metadata),
            "content": chunk,
            "company": company,
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat(),
        })

    faiss.write_index(self.index, INDEX_PATH)
    save_object(self.metadata, METADATA_PATH)
    return {"success": True, "chunks": len(chunks)}

def retrieve(self, query: str, top_k: int = 5, companies: List[str] = None):
    if self.index.ntotal == 0:
        return []
    q_vec = self.model.encode([query])
    filtered_indices = [
        i for i, m in enumerate(self.metadata)
        if not companies or m["company"] in companies
    ]
    if not filtered_indices:
        return []

    filtered_vectors = np.array([
        self.model.encode([self.metadata[i]['content']])[0]
        for i in filtered_indices
    ])
    temp_index = faiss.IndexFlatL2(384)
    temp_index.add(filtered_vectors.astype("float32"))
    D, I = temp_index.search(np.array(q_vec).astype("float32"), min(top_k, temp_index.ntotal))
    return [self.metadata[filtered_indices[i]] for i in I[0]]

def generate_with_gemini(self, prompt: str, max_tokens: int = 500):
    for model_name in GEMINI_MODELS:
        try:
            resp = self.gemini_clients[model_name].generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": 0.3},
            )
            return resp.text, model_name
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue
            return f"❌ Gemini error: {str(e)}", model_name
    return "❌ All Gemini models failed.", "none"

def extract_financials(self, chunks: List[str], company: str) -> List[Dict[str, Any]]:
    records = []
    for batch_start in range(0, len(chunks), 20):
        batch = chunks[batch_start:batch_start+20]
        context = "\n".join(batch)
        prompt = f"""

Extract financial metrics from the following text and return JSON only. Required fields: Company, Quarter, Revenue, OperatingIncome, OperatingMargin, NetIncome, EPS, ComparableEPS.

TEXT: {context}

JSON: """ raw, _ = self.generate_with_gemini(prompt, max_tokens=400) try: parsed = json.loads(raw) if isinstance(parsed, list): for r in parsed: r["Company"] = company records.extend(parsed) except Exception: st.warning(f"⚠️ Batch {batch_start}-{batch_start+20} could not be parsed.") return records

=========================

INIT BOT

=========================

@st.cache_resource def get_bot(): return FinancialRAGBot( load_embedding_model(), load_faiss_index(), load_metadata_cached(), load_embedding_cache_cached(), initialize_gemini_clients() )

bot = get_bot() if "history" not in st.session_state: st.session_state.history = [] if "financials" not in st.session_state: st.session_state.financials = []

=========================

SIDEBAR

=========================

st.sidebar.title("⚙️ Document Management") uploaded = st.sidebar.file_uploader("📂 Upload Financial PDF", type="pdf") company_tag = st.sidebar.text_input("🏷️ Company Name")

if uploaded: with st.spinner("🔄 Processing PDF..."): result = bot.process_pdf(uploaded.read(), company=company_tag or uploaded.name) if result["success"]: st.sidebar.success(f"✅ Added {result['chunks']} chunks from '{uploaded.name}'") st.rerun() else: st.sidebar.error(f"❌ {result['error']}")

if st.sidebar.button("🧹 Reset Index"): for f in [INDEX_PATH, METADATA_PATH, EMBEDDING_CACHE_PATH]: if os.path.exists(f): os.remove(f) st.cache_resource.clear(); st.cache_data.clear(); st.session_state.clear() st.rerun()

st.sidebar.subheader("Processed Documents") if bot.metadata: st.sidebar.json({"Companies": sorted(list(set([m['company'] for m in bot.metadata])))}) else: st.sidebar.info("No documents processed yet.")

=========================

MAIN UI

=========================

st.title("📊 Financial Report RAG Bot") st.markdown("Upload financial PDFs and ask questions. Gemini will summarize with context and source citation.")

companies = sorted(list(set(m['company'] for m in bot.metadata))) selected_companies = st.multiselect("Filter by Company:", options=companies, default=companies)

query = st.text_input("💬 Ask your financial question:")

if st.button("Get Answer"): if not query: st.warning("Please enter a question.") elif not bot.metadata: st.warning("Please upload and process at least one document first.") else: with st.spinner("Thinking..."): chunks = bot.retrieve(query, top_k=5, companies=selected_companies) if not chunks: answer = "I couldn't find relevant information." st.session_state.history.append((query, answer, [])) else: context = "\n\n".join([f"--- {c['company']} ---\n{c['content']}" for c in chunks]) prompt = f""" You are a financial assistant. Use the context to answer the question. Highlight numbers and percentages in bold. Always cite company sources.

CONTEXT: {context}

QUESTION: {query} ANSWER: """ answer, _ = bot.generate_with_gemini(prompt) st.session_state.history.append((query, highlight_numbers(answer), chunks))

=========================

FINANCIAL EXTRACTION

=========================

st.markdown("---") st.subheader("📑 Extract Financials")

if st.button("Extract Financials"): if not bot.metadata: st.warning("Upload and process at least one document first.") else: with st.spinner("🔎 Extracting structured financials..."): all_records = [] for company in companies: chunks = [m['content'] for m in bot.metadata if m['company'] == company] recs = bot.extract_financials(chunks, company) all_records.extend(recs) if all_records: df = pd.DataFrame(all_records) st.session_state.financials = df st.success("✅ Financials extracted!") st.dataframe(df) csv = df.to_csv(index=False).encode("utf-8") st.download_button("📥 Download CSV", data=csv, file_name="financials.csv", mime="text/csv") else: st.error("❌ No financials could be extracted.")

=========================

CHAT HISTORY

=========================

st.markdown("---") st.subheader("🗣️ Chat History") for q, a, srcs in reversed(st.session_state.history): st.markdown(f"You: {q}") st.markdown(f"Bot: {a}") if srcs: with st.expander("📚 Sources"): for c in srcs: st.markdown(f"- {c['company']}") st.markdown("---")

