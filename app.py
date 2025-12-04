import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import re
import json
from datetime import datetime
import hashlib
from typing import List, Dict, Any, Optional
import requests
import time
from dataclasses import dataclass
from enum import Enum

# -----------------
# CONFIG
# -----------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
EMBEDDING_VECTORS_PATH = "financial_index/vectors.pkl"  # fallback when faiss unavailable
EXTRACTION_HISTORY_PATH = "financial_index/extraction_history.pkl"

HF_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
}

CHUNK_SIZE = 800
OVERLAP = 150
MAX_RETRIES = 3
RETRY_DELAY = 2

os.makedirs("financial_index", exist_ok=True)

# -----------------
# Robust imports / fallbacks
# -----------------
# langchain text splitter fallback
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter as RecursiveCharacterTextSplitter
    except Exception:
        # Minimal local splitter fallback
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str) -> List[str]:
                # simple paragraph-based then sliding window
                paras = [p.strip() for p in text.split('\n\n') if p.strip()]
                chunks = []
                for p in paras:
                    start = 0
                    while start < len(p):
                        end = start + self.chunk_size
                        chunks.append(p[start:end])
                        start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
                if not chunks and text:
                    # fallback full text
                    return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
                return chunks

# faiss fallback
_FAISS_AVAILABLE = True
try:
    import faiss
except Exception:
    _FAISS_AVAILABLE = False

# -----------------
# Simple index fallback (when faiss not available)
# -----------------
class SimpleIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
    def add(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.vectors = np.vstack([self.vectors, arr])
    def search(self, query: np.ndarray, k: int):
        # query shape (1, dim) expected
        if self.vectors.shape[0] == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = np.asarray(query, dtype="float32")
        diffs = self.vectors - q
        dists = np.sum(diffs * diffs, axis=1)
        idx = np.argsort(dists)[:k]
        return np.array([dists[idx]]), np.array([idx])

# -----------------
# Data models
# -----------------
@dataclass
class FinancialMetrics:
    company: str
    period: str
    period_end_date: str
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    operating_margin: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    diluted_eps: Optional[float] = None
    gross_profit: Optional[float] = None
    gross_margin: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    cash_flow_operations: Optional[float] = None
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}
    def validate(self) -> List[str]:
        errors = []
        if not self.company or not self.period:
            errors.append("Missing company or period")
        if self.period_end_date:
            try:
                datetime.strptime(self.period_end_date, "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date: {self.period_end_date}")
        for field in ["revenue", "operating_income", "net_income", "total_assets"]:
            val = getattr(self, field)
            if val is not None and (val < 0 or val > 2_000_000):
                errors.append(f"{field} out of range: {val}")
        for margin_field in ["operating_margin", "gross_margin"]:
            val = getattr(self, margin_field)
            if val is not None and not (0 <= val <= 100):
                errors.append(f"{margin_field} should be 0-100%: {val}")
        if self.operating_income and self.revenue and self.operating_margin:
            calc_margin = (self.operating_income / self.revenue) * 100
            if abs(calc_margin - self.operating_margin) > 2.0:
                errors.append(f"Operating margin mismatch: stated {self.operating_margin}%, calc {calc_margin:.1f}%")
        return errors

class ExtractionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

@dataclass
class ExtractionResult:
    metrics: Optional[FinancialMetrics]
    status: ExtractionStatus
    errors: List[str]
    warnings: List[str]
    model_used: str
    timestamp: str
    chunks_used: List[str]
    confidence_score: float

# -----------------
# Utilities and clients
# -----------------
class HuggingFaceClient:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self.base_url = "https://api-inference.huggingface.co/models/"
    def generate(self, model_name: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
        url = self.base_url + model_name
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=60)
                if response.status_code == 503:
                    try:
                        st.info(f"⏳ Model loading... (attempt {attempt + 1}/{MAX_RETRIES})")
                    except Exception:
                        pass
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    return str(result)
                else:
                    try:
                        st.warning(f"⚠️ API Error {response.status_code}: {response.text}")
                    except Exception:
                        pass
                    return None
            except Exception as e:
                try:
                    st.warning(f"⚠️ Request failed (attempt {attempt + 1}): {e}")
                except Exception:
                    pass
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        return None

# -----------------
# Main RAG Bot
# -----------------
class AdvancedFinancialRAGBot:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_client = HuggingFaceClient(hf_token)
        self._embedding_model = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.extraction_history: List[Dict[str, Any]] = []
        self.index = None
        self.vectors_fallback = []
        self.dim = None
        self._load_state()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            # cache resource to avoid reloads in Streamlit
            try:
                @st.cache_resource
                def _load_model(name: str):
                    return SentenceTransformer(name)
                self._embedding_model = _load_model(EMBEDDING_MODEL)
            except Exception:
                # last-resort load (non-cached)
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _load_state(self):
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                self.embedding_cache = pickle.load(f)
        if os.path.exists(EXTRACTION_HISTORY_PATH):
            with open(EXTRACTION_HISTORY_PATH, "rb") as f:
                self.extraction_history = pickle.load(f)
        if _FAISS_AVAILABLE and os.path.exists(INDEX_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                self.dim = self.index.d
            except Exception:
                self.index = None
        else:
            # load fallback vectors
            if os.path.exists(EMBEDDING_VECTORS_PATH):
                with open(EMBEDDING_VECTORS_PATH, "rb") as f:
                    self.vectors_fallback = pickle.load(f)
                if len(self.vectors_fallback) > 0:
                    self.vectors_fallback = [np.asarray(v, dtype="float32") for v in self.vectors_fallback]
                    self.dim = int(self.vectors_fallback[0].shape[0])
                    self.index = SimpleIndex(self.dim)
                    for v in self.vectors_fallback:
                        self.index.add(v)

    def _save_state(self):
        if _FAISS_AVAILABLE and self.index is not None and hasattr(self.index, 'is_trained'):
            try:
                faiss.write_index(self.index, INDEX_PATH)
            except Exception:
                pass
        else:
            # save vectors fallback
            try:
                if hasattr(self.index, 'vectors'):
                    vectors = [v for v in self.index.vectors]
                    with open(EMBEDDING_VECTORS_PATH, "wb") as f:
                        pickle.dump(vectors, f)
            except Exception:
                pass
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(self.embedding_cache, f)
        with open(EXTRACTION_HISTORY_PATH, "wb") as f:
            pickle.dump(self.extraction_history, f)

    def add_document(self, uploaded_file, company: str, report_type: str = "earnings") -> Dict[str, Any]:
        try:
            # read bytes and extract text from BytesIO to avoid stream position issues
            raw = uploaded_file.read()
            text = extract_text(BytesIO(raw))
            if not text or len(text) < 50:
                return {"success": False, "error": "Document appears empty or invalid"}
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
            chunks = splitter.split_text(text)
            if not chunks:
                return {"success": False, "error": "No text chunks produced"}
            if self.index is None:
                # initialize index
                dim = self.embedding_model.get_sentence_embedding_dimension()
                self.dim = dim
                if _FAISS_AVAILABLE:
                    self.index = faiss.IndexFlatL2(dim)
                else:
                    self.index = SimpleIndex(dim)
            new_chunks = []
            new_hashes = []
            for chunk in chunks:
                h = hashlib.sha256(chunk.encode()).hexdigest()
                if h not in self.embedding_cache:
                    new_chunks.append(chunk)
                    new_hashes.append(h)
            if new_chunks:
                with st.spinner(f"Embedding {len(new_chunks)} new chunks..."):
                    embeddings = self.embedding_model.encode(new_chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
                for chunk, emb, h in zip(new_chunks, embeddings, new_hashes):
                    self.embedding_cache[h] = emb
                    # add to index
                    if _FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
                        self.index.add(np.array([emb], dtype="float32"))
                    else:
                        self.index.add(emb)
                    self.metadata.append({
                        "company": company,
                        "content": chunk,
                        "hash": h,
                        "report_type": report_type,
                        "added_date": datetime.now().isoformat()
                    })
            self._save_state()
            return {"success": True, "chunks_added": len(new_chunks), "total_chunks": len(chunks), "duplicates_skipped": len(chunks) - len(new_chunks)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def semantic_search(self, query: str, company: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.index or not self.metadata:
            return []
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        k = min(top_k * 3, len(self.metadata))
        if _FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            distances, indices = self.index.search(np.array([query_emb], dtype="float32"), k)
        else:
            distances, indices = self.index.search(np.array([query_emb], dtype="float32"), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if company is None or meta["company"] == company:
                results.append({
                    "content": meta["content"],
                    "company": meta["company"],
                    "distance": float(dist),
                    "similarity": float(1 / (1 + dist)) if dist >= 0 else 0.0
                })
                if len(results) >= top_k:
                    break
        return results

    def extract_json_robust(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        patterns = [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})']
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except Exception:
                    continue
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return None

    def build_extraction_prompt(self, chunks: List[str], company: str) -> str:
        context = "\n\n---\n\n".join(chunks[:5])
        prompt = f"""<|system|>
You are a financial data extraction expert. Extract structured financial metrics from earnings reports.

<|user|>
TASK: Extract financial data for {company} from the text below.

OUTPUT: Return ONLY a valid JSON object with this exact structure:
{{
  "company": "{company}",
  "period": "Q1 2024",
  "period_end_date": "2024-03-31",
  "revenue": 52500.0,
  "operating_income": 18900.0,
  "operating_margin": 36.0,
  "net_income": 21000.0,
  "eps": 2.03,
  "diluted_eps": 2.03,
  "gross_profit": 35000.0,
  "gross_margin": 66.7
}}

RULES:
1. All monetary values in millions USD
2. Convert: "$12.5B" → 12500, "$3,200M" → 3200
3. Margins as percentages (0-100)
4. Use diluted_eps if available
5. Use null for missing values
6. Extract most recent quarter/year only
7. Dates as YYYY-MM-DD

DOCUMENT:
{context[:8000]}

<|assistant|>
{{"""
        return prompt

    def calculate_confidence(self, metrics: FinancialMetrics, chunks: List[str]) -> float:
        score = 0.0
        if metrics.revenue and metrics.net_income and metrics.period:
            score += 40
        optional_fields = [metrics.operating_income, metrics.eps, metrics.gross_profit]
        score += (sum(1 for f in optional_fields if f is not None) / len(optional_fields)) * 20
        errors = metrics.validate()
        if len(errors) == 0:
            score += 20
        revenue_mentions = sum(1 for c in chunks if "revenue" in c.lower() or "sales" in c.lower())
        if revenue_mentions >= 2:
            score += 20
        return min(score, 100.0)

    def extract_financials(self, company: str, model_name: str = "mistral-7b") -> ExtractionResult:
        search_query = f"{company} revenue operating income net income earnings per share quarterly fiscal year financial results"
        chunks = self.semantic_search(search_query, company=company, top_k=10)
        if not chunks:
            return ExtractionResult(metrics=None, status=ExtractionStatus.FAILED, errors=["No documents found for this company"], warnings=[], model_used=model_name, timestamp=datetime.now().isoformat(), chunks_used=[], confidence_score=0.0)
        chunk_texts = [c["content"] for c in chunks]
        prompt = self.build_extraction_prompt(chunk_texts, company)
        model_path = HF_MODELS.get(model_name) or list(HF_MODELS.values())[0]
        with st.spinner(f"🤖 Extracting with {model_name}..."):
            response = self.hf_client.generate(model_path, prompt, max_tokens=1000, temperature=0.1)
        if not response:
            return ExtractionResult(metrics=None, status=ExtractionStatus.FAILED, errors=["LLM generation failed"], warnings=[], model_used=model_name, timestamp=datetime.now().isoformat(), chunks_used=chunk_texts, confidence_score=0.0)
        data = self.extract_json_robust(response)
        if not data:
            return ExtractionResult(metrics=None, status=ExtractionStatus.FAILED, errors=["Failed to parse JSON from LLM response"], warnings=[f"Raw response: {response[:500]}"], model_used=model_name, timestamp=datetime.now().isoformat(), chunks_used=chunk_texts, confidence_score=0.0)
        try:
            metrics = FinancialMetrics(
                company=data.get("company", company),
                period=data.get("period", "Unknown"),
                period_end_date=data.get("period_end_date", ""),
                revenue=data.get("revenue"),
                operating_income=data.get("operating_income"),
                operating_margin=data.get("operating_margin"),
                net_income=data.get("net_income"),
                eps=data.get("eps"),
                diluted_eps=data.get("diluted_eps"),
                gross_profit=data.get("gross_profit"),
                gross_margin=data.get("gross_margin"),
                total_assets=data.get("total_assets"),
                total_equity=data.get("total_equity"),
                cash_flow_operations=data.get("cash_flow_operations")
            )
        except Exception as e:
            return ExtractionResult(metrics=None, status=ExtractionStatus.FAILED, errors=[f"Failed to create metrics object: {str(e)}"], warnings=[], model_used=model_name, timestamp=datetime.now().isoformat(), chunks_used=chunk_texts, confidence_score=0.0)
        validation_errors = metrics.validate()
        warnings = []
        if not metrics.operating_income:
            warnings.append("Operating income not found")
        if not metrics.eps:
            warnings.append("EPS not found")
        confidence = self.calculate_confidence(metrics, chunk_texts)
        status = ExtractionStatus.SUCCESS
        if validation_errors:
            status = ExtractionStatus.PARTIAL if confidence > 50 else ExtractionStatus.FAILED
        result = ExtractionResult(metrics=metrics, status=status, errors=validation_errors, warnings=warnings, model_used=model_name, timestamp=datetime.now().isoformat(), chunks_used=chunk_texts[:3], confidence_score=confidence)
        self.extraction_history.append({"company": company, "timestamp": result.timestamp, "status": status.value, "confidence": confidence, "model": model_name})
        self._save_state()
        return result

    def get_companies(self) -> List[str]:
        return sorted(set(m["company"] for m in self.metadata))

    def compare_companies(self, companies: List[str]) -> pd.DataFrame:
        results = []
        for company in companies:
            result = self.extract_financials(company)
            if result.metrics:
                row = result.metrics.to_dict()
                row["confidence"] = result.confidence_score
                row["status"] = result.status.value
                results.append(row)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        cache_size = 0.0
        try:
            cache_size = sum(len(pickle.dumps(v)) for v in self.embedding_cache.values()) / 1024 / 1024
        except Exception:
            cache_size = 0.0
        return {"total_documents": len(set(m["hash"] for m in self.metadata)), "total_chunks": len(self.metadata), "companies": len(self.get_companies()), "extractions_performed": len(self.extraction_history), "cache_size_mb": cache_size}

# -----------------
# Streamlit UI
# -----------------
st.set_page_config(page_title="🚀 Advanced Financial RAG Bot", layout="wide")

st.markdown("""
<style>
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0; }
    .success-box { background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Financial RAG Bot")
st.markdown("**Powered by Free HuggingFace Models** 🤗")
st.caption("Extract, analyze, and compare financial metrics from earnings reports using state-of-the-art AI")

with st.sidebar:
    st.header("⚙️ Configuration")
    hf_token = st.text_input("🤗 HuggingFace API Token (Optional)", type="password", help="Get free token at https://huggingface.co/settings/tokens")
    st.divider()
    model_choice = st.selectbox("🤖 Select LLM Model", options=list(HF_MODELS.keys()), index=0, help="Mistral-7B recommended for best results")
    st.divider()
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **{model_choice}**
    • Free via HuggingFace API
    • No GPU required
    • Rate limits apply
    """)

if 'bot' not in st.session_state:
    st.session_state.bot = AdvancedFinancialRAGBot(hf_token if hf_token else None)
bot = st.session_state.bot

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Extract", "🔍 Compare", "📈 Analytics"])

with tab1:
    st.header("📂 Document Management")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type=["pdf"], help="Upload earnings reports, 10-K, 10-Q, or other financial documents")
    with col2:
        if uploaded_file:
            company_name = st.text_input("Company Name", value=uploaded_file.name.split(".")[0].replace("_", " ").title(), help="Enter the company name exactly")
            report_type = st.selectbox("Report Type", ["Earnings", "10-K", "10-Q", "Annual Report", "Other"])
    if uploaded_file and company_name:
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                result = bot.add_document(uploaded_file, company_name, report_type.lower())
            if result.get("success"):
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                c1, c2, c3 = st.columns(3)
                c1.metric("New Chunks", result["chunks_added"])
                c2.metric("Total Chunks", result["total_chunks"])
                c3.metric("Duplicates", result["duplicates_skipped"])
            else:
                st.error(f"❌ Error: {result.get('error')}")
    if bot.metadata:
        st.divider()
        st.subheader("📚 Indexed Companies")
        companies = bot.get_companies()
        st.write(f"**{len(companies)} companies** in database:")
        st.write(", ".join(companies))

with tab2:
    st.header("📊 Extract Financial Metrics")
    companies = bot.get_companies()
    if not companies:
        st.warning("⚠️ No documents uploaded yet. Go to the Upload tab to add financial reports.")
    else:
        selected_company = st.selectbox("Select Company", companies)
        if st.button("🎯 Extract Financials", type="primary"):
            result = bot.extract_financials(selected_company, model_choice)
            col1, col2 = st.columns([3, 1])
            with col2:
                if result.status == ExtractionStatus.SUCCESS:
                    st.success(f"✅ {result.status.value.upper()}")
                elif result.status == ExtractionStatus.PARTIAL:
                    st.warning(f"⚠️ {result.status.value.upper()}")
                else:
                    st.error(f"❌ {result.status.value.upper()}")
                st.metric("Confidence", f"{result.confidence_score:.0f}%")
                st.caption(f"Model: {result.model_used}")
            with col1:
                if result.metrics:
                    df = pd.DataFrame([result.metrics.to_dict()])
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", data=csv, file_name=f"{selected_company}_financials_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            if result.errors:
                st.error("**Validation Errors:**")
                for error in result.errors:
                    st.write(f"• {error}")
            if result.warnings:
                st.warning("**Warnings:**")
                for warning in result.warnings:
                    st.write(f"• {warning}")
            with st.expander("📄 View Source Chunks"):
                for i, chunk in enumerate(result.chunks_used, 1):
                    st.text_area(f"Chunk {i}", chunk, height=100)

with tab3:
    st.header("🔍 Compare Companies")
    companies = bot.get_companies()
    if len(companies) < 2:
        st.info("ℹ️ Upload at least 2 companies to enable comparison")
    else:
        selected_companies = st.multiselect("Select Companies to Compare", companies, default=companies[:min(3, len(companies))])
        if selected_companies and st.button("📊 Compare", type="primary"):
            with st.spinner("Extracting metrics for all companies..."):
                comparison_df = bot.compare_companies(selected_companies)
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
                st.subheader("📈 Visual Comparison")
                c1, c2 = st.columns(2)
                with c1:
                    if "revenue" in comparison_df.columns:
                        st.bar_chart(comparison_df.set_index("company")["revenue"])
                        st.caption("Revenue (Millions USD)")
                with c2:
                    if "operating_margin" in comparison_df.columns:
                        st.bar_chart(comparison_df.set_index("company")["operating_margin"])
                        st.caption("Operating Margin (%)")
                csv = comparison_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Comparison CSV", data=csv, file_name=f"company_comparison_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            else:
                st.error("Failed to extract metrics for comparison")

with tab4:
    st.header("📈 System Analytics")
    stats = bot.get_statistics()
    a, b, c, d = st.columns(4)
    a.metric("📄 Documents", stats["total_documents"]) 
    b.metric("🧩 Chunks", stats["total_chunks"]) 
    c.metric("🏢 Companies", stats["companies"]) 
    d.metric("🔍 Extractions", stats["extractions_performed"]) 
    st.divider()
    if bot.extraction_history:
        st.subheader("📊 Extraction History")
        history_df = pd.DataFrame(bot.extraction_history)
        st.dataframe(history_df, use_container_width=True)
        if not history_df.empty:
            success_rate = (history_df["status"] == "success").sum() / len(history_df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    st.divider()
    st.subheader("🔧 System Information")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    st.write(f"**Cache Size:** {stats['cache_size_mb']:.2f} MB")
    st.write(f"**Index Type:** {'FAISS IndexFlatL2' if _FAISS_AVAILABLE else 'SimpleIndex (fallback)'}")
    st.divider()
    st.subheader("⚠️ Danger Zone")
    x1, x2 = st.columns(2)
    with x1:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all documents and extractions"):
                try:
                    for file_path in [INDEX_PATH, METADATA_PATH, EMBEDDING_CACHE_PATH, EXTRACTION_HISTORY_PATH, EMBEDDING_VECTORS_PATH]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    st.session_state.bot = AdvancedFinancialRAGBot(hf_token if hf_token else None)
                    st.success("✅ All data cleared successfully")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {e}")
    with x2:
        if st.button("💾 Export All Data"):
            try:
                export_data = {"metadata": bot.metadata, "extraction_history": bot.extraction_history, "statistics": stats}
                export_json = json.dumps(export_data, indent=2)
                st.download_button("📥 Download Export", data=export_json, file_name=f"financial_rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            except Exception as e:
                st.error(f"❌ Error exporting data: {e}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Advanced Financial RAG Bot</strong> | Powered by HuggingFace 🤗</p>
    <p style='font-size: 0.9em;'><em>Built with Streamlit • FAISS (if available) • Sentence Transformers • Free LLMs</em></p>
</div>
""", unsafe_allow_html=True)
