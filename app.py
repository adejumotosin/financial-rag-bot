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

# ================= CONFIG =================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
EMBEDDING_VECTORS_PATH = "financial_index/vectors.pkl"
EXTRACTION_HISTORY_PATH = "financial_index/extraction_history.pkl"

GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-70b": "llama-3.1-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "gemma-7b": "gemma-7b-it",
}

CHUNK_SIZE = 800
OVERLAP = 150
MAX_RETRIES = 3
RETRY_DELAY = 2

os.makedirs("financial_index", exist_ok=True)

# ================= Robust imports / fallbacks =================
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter as RecursiveCharacterTextSplitter
    except Exception:
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str) -> List[str]:
                paras = [p.strip() for p in text.split('\n\n') if p.strip()]
                chunks = []
                for p in paras:
                    start = 0
                    while start < len(p):
                        end = start + self.chunk_size
                        chunks.append(p[start:end])
                        start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
                if not chunks and text:
                    return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
                return chunks

_FAISS_AVAILABLE = True
try:
    import faiss
except Exception:
    _FAISS_AVAILABLE = False

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
        if self.vectors.shape[0] == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = np.asarray(query, dtype="float32")
        diffs = self.vectors - q
        dists = np.sum(diffs * diffs, axis=1)
        idx = np.argsort(dists)[:k]
        return np.array([dists[idx]]), np.array([idx])

# ================= Data models =================
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

# ================= Groq Client =================
class GroqClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, model_name: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
        if not self.api_key:
            return None
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 429:
                    try:
                        st.warning(f"⏳ Rate limited... (attempt {attempt + 1}/{MAX_RETRIES})")
                    except Exception:
                        pass
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    return None
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

# ================= Main RAG Bot =================
class AdvancedFinancialRAGBot:
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_client = GroqClient(groq_api_key)
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
            try:
                @st.cache_resource
                def _load_model(name: str):
                    return SentenceTransformer(name)
                self._embedding_model = _load_model(EMBEDDING_MODEL)
            except Exception:
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
            raw = uploaded_file.read()
            text = extract_text(BytesIO(raw))
            if not text or len(text) < 50:
                return {"success": False, "error": "Document appears empty or invalid"}
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
            chunks = splitter.split_text(text)
            if not chunks:
                return {"success": False, "error": "No text chunks produced"}
            
            if self.index is None:
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
            return {
                "success": True,
                "chunks_added": len(new_chunks),
                "total_chunks": len(chunks),
                "duplicates_skipped": len(chunks) - len(new_chunks)
            }
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
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try markdown code blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except Exception:
                    continue
        
        # Try finding JSON boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        
        return None
    
    def build_extraction_prompt(self, chunks: List[str], company: str) -> str:
        """IMPROVED: More explicit instructions and examples"""
        context = "\n\n---CHUNK---\n\n".join(chunks[:8])  # Use more chunks
        
        prompt = f"""You are a financial data extraction expert. Extract financial metrics for {company} from the earnings report below.

**CRITICAL INSTRUCTIONS:**
1. Look for tables labeled "Consolidated Statements of Income" or "Income Statement"
2. Find the MOST RECENT quarter/period (usually rightmost column in tables)
3. Convert ALL values to millions USD:
   - "$12.5 billion" → 12500
   - "$3,200 million" → 3200
   - "3.2B" → 3200
4. For margins, calculate if not stated: (operating_income / revenue) × 100
5. Look for these EXACT terms or close variants:
   - Revenue: "Net revenues", "Total revenues", "Net sales"
   - Operating Income: "Operating income", "Income from operations"
   - Net Income: "Net income", "Net income attributable to shareholders"
   - EPS: "Basic earnings per share", "Earnings per share - basic"
   - Diluted EPS: "Diluted earnings per share", "Earnings per share - diluted"

**OUTPUT FORMAT (JSON only, no explanation):**
{{
  "company": "{company}",
  "period": "Q3 2025",
  "period_end_date": "2025-09-30",
  "revenue": 11900.0,
  "operating_income": 3200.0,
  "operating_margin": 26.9,
  "net_income": 2900.0,
  "eps": 0.67,
  "diluted_eps": 0.67,
  "gross_profit": 7100.0,
  "gross_margin": 59.7
}}

**DOCUMENT CHUNKS:**
{context[:12000]}

Return ONLY the JSON object:"""
        return prompt
    
    def calculate_confidence(self, metrics: FinancialMetrics, chunks: List[str]) -> float:
        """IMPROVED: Better confidence scoring"""
        score = 0.0
        
        # Core fields (50 points)
        if metrics.revenue and metrics.net_income and metrics.period:
            score += 30
        if metrics.revenue:
            score += 10
        if metrics.net_income:
            score += 10
        
        # Important fields (30 points)
        if metrics.operating_income:
            score += 15
        if metrics.eps or metrics.diluted_eps:
            score += 15
        
        # Validation (20 points)
        errors = metrics.validate()
        if len(errors) == 0:
            score += 20
        elif len(errors) <= 2:
            score += 10
        
        return min(score, 100.0)
    
    def extract_financials(self, company: str, model_name: str = "llama-3.3-70b") -> ExtractionResult:
        """IMPROVED: Multi-query approach for better retrieval"""
        
        # Use multiple targeted queries
        queries = [
            f"{company} consolidated statements of income revenue net income",
            f"{company} operating income earnings per share EPS",
            f"{company} quarterly results Q1 Q2 Q3 Q4 fiscal year"
        ]
        
        all_chunks = []
        seen_hashes = set()
        
        for query in queries:
            chunks = self.semantic_search(query, company=company, top_k=5)
            for chunk in chunks:
                h = hashlib.sha256(chunk["content"].encode()).hexdigest()
                if h not in seen_hashes:
                    all_chunks.append(chunk)
                    seen_hashes.add(h)
        
        if not all_chunks:
            return ExtractionResult(
                metrics=None,
                status=ExtractionStatus.FAILED,
                errors=["No documents found for this company"],
                warnings=[],
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                chunks_used=[],
                confidence_score=0.0
            )
        
        chunk_texts = [c["content"] for c in all_chunks[:10]]  # Use top 10
        prompt = self.build_extraction_prompt(chunk_texts, company)
        model_id = GROQ_MODELS.get(model_name) or list(GROQ_MODELS.values())[0]
        
        with st.spinner(f"🤖 Extracting with {model_name} via Groq..."):
            response = self.groq_client.generate(model_id, prompt, max_tokens=1500, temperature=0.05)
        
        if not response:
            return ExtractionResult(
                metrics=None,
                status=ExtractionStatus.FAILED,
                errors=["LLM generation failed - please check your API key"],
                warnings=[],
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                chunks_used=chunk_texts,
                confidence_score=0.0
            )
        
        # Debug: show raw response
        with st.expander("🔍 Debug: Raw LLM Response"):
            st.code(response, language="json")
        
        data = self.extract_json_robust(response)
        if not data:
            return ExtractionResult(
                metrics=None,
                status=ExtractionStatus.FAILED,
                errors=["Failed to parse JSON from LLM response"],
                warnings=[f"Raw response: {response[:500]}"],
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                chunks_used=chunk_texts,
                confidence_score=0.0
            )
        
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
            return ExtractionResult(
                metrics=None,
                status=ExtractionStatus.FAILED,
                errors=[f"Failed to create metrics object: {str(e)}"],
                warnings=[],
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                chunks_used=chunk_texts,
                confidence_score=0.0
            )
        
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
        elif confidence < 50:
            status = ExtractionStatus.PARTIAL
        
        result = ExtractionResult(
            metrics=metrics,
            status=status,
            errors=validation_errors,
            warnings=warnings,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            chunks_used=chunk_texts[:3],
            confidence_score=confidence
        )
        
        self.extraction_history.append({
            "company": company,
            "timestamp": result.timestamp,
            "status": status.value,
            "confidence": confidence,
            "model": model_name
        })
        self._save_state()
        return result
    
    def get_companies(self) -> List[str]:
        return sorted(set(m["company"] for m in self.metadata))
    
    def compare_companies(self, companies: List[str], model_name: str = "llama-3.3-70b") -> pd.DataFrame:
        results = []
        for company in companies:
            result = self.extract_financials(company, model_name)
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
        return {
            "total_documents": len(set(m["hash"] for m in self.metadata)),
            "total_chunks": len(self.metadata),
            "companies": len(self.get_companies()),
            "extractions_performed": len(self.extraction_history),
            "cache_size_mb": cache_size
        }

# ================= Streamlit UI =================
st.set_page_config(page_title="🚀 Advanced Financial RAG Bot", layout="wide")

st.markdown("""
<style>
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0; }
    .success-box { background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; }
    .warning-box { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Financial RAG Bot")
st.markdown("⚡ Powered by Groq AI - Lightning Fast Inference")
st.caption("Extract, analyze, and compare financial metrics from earnings reports using state-of-the-art AI")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Get your free API key at https://console.groq.com"
    )
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key to use extraction features")
        st.markdown("[Get Free API Key →](https://console.groq.com)")
    
    st.divider()
    model_choice = st.selectbox(
        "🤖 Select LLM Model",
        options=list(GROQ_MODELS.keys()),
        index=0,
        help="Llama 3.3 70B recommended for best accuracy"
    )
    
    st.divider()
    st.markdown("### 📊 Model Info")
    st.info(f"""
**{model_choice}**
• Ultra-fast inference via Groq
• Free tier: 30 req/min
• Best-in-class accuracy
""")

if 'bot' not in st.session_state:
    st.session_state.bot = AdvancedFinancialRAGBot(groq_api_key if groq_api_key else None)
else:
    st.session_state.bot.groq_client = GroqClient(groq_api_key if groq_api_key else None)

bot = st.session_state.bot

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Extract", "🔍 Compare", "📈 Analytics"])

with tab1:
    st.header("📂 Document Management")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Financial Report (PDF)",
            type=["pdf"],
            help="Upload earnings reports, 10-K, 10-Q, or other financial documents"
        )
    
    with col2:
        if uploaded_file:
            company_name = st.text_input(
                "Company Name",
                value=uploaded_file.name.split(".")[0].replace("_", " ").title(),
                help="Enter the company name exactly"
            )
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
        st.write(f"{len(companies)} companies in database:")
        st.write(", ".join(companies))

with tab2:
    st.header("📊 Extract Financial Metrics")
    companies = bot.get_companies()
    
    if not companies:
        st.warning("⚠️ No documents uploaded yet. Go to the Upload tab to add financial reports.")
    elif not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use extraction features.")
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
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"{selected_company}_financials_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            if result.errors:
                st.error("Validation Errors:")
                for error in result.errors:
                    st.write(f"• {error}")
            
            if result.warnings:
                st.warning("Warnings:")
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
    elif not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use comparison features.")
    else:
        selected_companies = st.multiselect(
            "Select Companies to Compare",
            companies,
            default=companies[:min(3, len(companies))]
        )
        
        if selected_companies and st.button("📊 Compare", type="primary"):
            with st.spinner("Extracting metrics for all companies..."):
                comparison_df = bot.compare_companies(selected_companies, model_choice)
            
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
                st.download_button(
                    "📥 Download Comparison CSV",
                    data=csv,
                    file_name=f"company_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
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
    st.write(f"**LLM Provider:** Groq (Ultra-fast inference)")
    
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
                    st.session_state.bot = AdvancedFinancialRAGBot(groq_api_key if groq_api_key else None)
                    st.success("✅ All data cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {e}")
    
    with x2:
        if st.button("💾 Export All Data"):
            try:
                export_data = {
                    "metadata": bot.metadata,
                    "extraction_history": bot.extraction_history,
                    "statistics": stats
                }
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    "📥 Download Export",
                    data=export_json,
                    file_name=f"financial_rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Error exporting data: {e}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Advanced Financial RAG Bot</strong> | Powered by Groq AI ⚡</p>
    <p style='font-size: 0.9em;'><em>Built with Streamlit • FAISS • Sentence Transformers • Groq LLMs</em></p>
    <p style='font-size: 0.8em; margin-top: 10px;'>Get your free Groq API key at <a href='https://console.groq.com' target='_blank'>console.groq.com</a></p>
</div>
""", unsafe_allow_html=True)