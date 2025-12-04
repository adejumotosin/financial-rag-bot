import streamlit as st
import numpy as np
import faiss
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import requests
import time
from dataclasses import dataclass
from enum import Enum

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
EXTRACTION_HISTORY_PATH = "financial_index/extraction_history.pkl"

# Free HuggingFace models (via Inference API)
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


# =========================
# DATA MODELS
# =========================
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
        
        # Required fields
        if not self.company or not self.period:
            errors.append("Missing company or period")
        
        # Date validation
        if self.period_end_date:
            try:
                datetime.strptime(self.period_end_date, "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date: {self.period_end_date}")
        
        # Range checks
        for field in ["revenue", "operating_income", "net_income", "total_assets"]:
            val = getattr(self, field)
            if val is not None and (val < 0 or val > 2_000_000):
                errors.append(f"{field} out of range: {val}")
        
        # Margin validation
        for margin_field in ["operating_margin", "gross_margin"]:
            val = getattr(self, margin_field)
            if val is not None and not (0 <= val <= 100):
                errors.append(f"{margin_field} should be 0-100%: {val}")
        
        # Cross-validation
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


# =========================
# HUGGINGFACE LLM CLIENT
# =========================
class HuggingFaceClient:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self.base_url = "https://api-inference.huggingface.co/models/"
    
    def generate(self, model_name: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Optional[str]:
        """Generate text using HuggingFace Inference API."""
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
                
                if response.status_code == 503:  # Model loading
                    st.info(f"⏳ Model loading... (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    return str(result)
                
                else:
                    st.warning(f"⚠️ API Error {response.status_code}: {response.text}")
                    return None
                    
            except Exception as e:
                st.warning(f"⚠️ Request failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        return None


# =========================
# ADVANCED FINANCIAL RAG BOT
# =========================
class AdvancedFinancialRAGBot:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_client = HuggingFaceClient(hf_token)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.metadata = []
        self.embedding_cache = {}
        self.extraction_history = []
        self.index = None
        
        os.makedirs("financial_index", exist_ok=True)
        self._load_state()
    
    def _load_state(self):
        """Load saved indices and metadata."""
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
        
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                self.embedding_cache = pickle.load(f)
        
        if os.path.exists(EXTRACTION_HISTORY_PATH):
            with open(EXTRACTION_HISTORY_PATH, "rb") as f:
                self.extraction_history = pickle.load(f)
        
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
    
    def _save_state(self):
        """Persist indices and metadata."""
        if self.index:
            faiss.write_index(self.index, INDEX_PATH)
        
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(self.embedding_cache, f)
        
        with open(EXTRACTION_HISTORY_PATH, "wb") as f:
            pickle.dump(self.extraction_history, f)
    
    def add_document(self, file, company: str, report_type: str = "earnings") -> Dict[str, Any]:
        """Process and index a financial document."""
        try:
            text = extract_text(file)
            if not text or len(text) < 100:
                return {"success": False, "error": "Document appears empty or invalid"}
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)
            
            if self.index is None:
                dim = self.embedding_model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dim)
            
            # Batch embedding for efficiency
            new_chunks = []
            new_hashes = []
            
            for chunk in chunks:
                h = hashlib.sha256(chunk.encode()).hexdigest()
                if h not in self.embedding_cache:
                    new_chunks.append(chunk)
                    new_hashes.append(h)
            
            if new_chunks:
                with st.spinner(f"Embedding {len(new_chunks)} new chunks..."):
                    embeddings = self.embedding_model.encode(
                        new_chunks,
                        batch_size=32,
                        show_progress_bar=True
                    )
                
                for chunk, embedding, h in zip(new_chunks, embeddings, new_hashes):
                    self.embedding_cache[h] = embedding
                    self.index.add(np.array([embedding], dtype="float32"))
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
        """Search for relevant chunks using semantic similarity."""
        if not self.index or not self.metadata:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding], dtype="float32"),
            min(top_k * 3, len(self.metadata))  # Over-fetch for filtering
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                if company is None or meta["company"] == company:
                    results.append({
                        "content": meta["content"],
                        "company": meta["company"],
                        "distance": float(dist),
                        "similarity": float(1 / (1 + dist))  # Convert to similarity score
                    })
                    if len(results) >= top_k:
                        break
        
        return results
    
    def extract_json_robust(self, text: str) -> Optional[dict]:
        """Robustly extract JSON from LLM output."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Extract from markdown code blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'  # Nested JSON
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Find first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        
        return None
    
    def build_extraction_prompt(self, chunks: List[str], company: str) -> str:
        """Build optimized prompt for financial extraction."""
        context = "\n\n---\n\n".join(chunks[:5])  # Top 5 chunks only
        
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
        """Calculate extraction confidence score."""
        score = 0.0
        
        # Has required fields
        if metrics.revenue and metrics.net_income and metrics.period:
            score += 40
        
        # Has optional fields
        optional_fields = [metrics.operating_income, metrics.eps, metrics.gross_profit]
        score += (sum(1 for f in optional_fields if f is not None) / len(optional_fields)) * 20
        
        # Cross-validation passed
        errors = metrics.validate()
        if len(errors) == 0:
            score += 20
        
        # Found in multiple chunks (verification)
        revenue_mentions = sum(1 for c in chunks if "revenue" in c.lower() or "sales" in c.lower())
        if revenue_mentions >= 2:
            score += 20
        
        return min(score, 100.0)
    
    def extract_financials(self, company: str, model_name: str = "mistral-7b") -> ExtractionResult:
        """Extract financial metrics using semantic search + LLM."""
        # Search for financial keywords
        search_query = f"{company} revenue operating income net income earnings per share quarterly fiscal year financial results"
        chunks = self.semantic_search(search_query, company=company, top_k=10)
        
        if not chunks:
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
        
        # Build prompt with top chunks
        chunk_texts = [c["content"] for c in chunks]
        prompt = self.build_extraction_prompt(chunk_texts, company)
        
        # Generate with LLM
        model_path = HF_MODELS.get(model_name)
        if not model_path:
            model_path = HF_MODELS["mistral-7b"]
        
        with st.spinner(f"🤖 Extracting with {model_name}..."):
            response = self.hf_client.generate(model_path, prompt, max_tokens=1000, temperature=0.1)
        
        if not response:
            return ExtractionResult(
                metrics=None,
                status=ExtractionStatus.FAILED,
                errors=["LLM generation failed"],
                warnings=[],
                model_used=model_name,
                timestamp=datetime.now().isoformat(),
                chunks_used=chunk_texts,
                confidence_score=0.0
            )
        
        # Parse JSON
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
        
        # Convert to FinancialMetrics
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
        
        # Validate
        validation_errors = metrics.validate()
        warnings = []
        
        # Check for missing optional fields
        if not metrics.operating_income:
            warnings.append("Operating income not found")
        if not metrics.eps:
            warnings.append("EPS not found")
        
        confidence = self.calculate_confidence(metrics, chunk_texts)
        
        status = ExtractionStatus.SUCCESS
        if validation_errors:
            status = ExtractionStatus.PARTIAL if confidence > 50 else ExtractionStatus.FAILED
        
        result = ExtractionResult(
            metrics=metrics,
            status=status,
            errors=validation_errors,
            warnings=warnings,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            chunks_used=chunk_texts[:3],  # Store top 3 for audit
            confidence_score=confidence
        )
        
        # Save to history
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
        """Get list of indexed companies."""
        return sorted(set(m["company"] for m in self.metadata))
    
    def compare_companies(self, companies: List[str]) -> pd.DataFrame:
        """Extract and compare metrics across multiple companies."""
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
        """Get RAG system statistics."""
        return {
            "total_documents": len(set(m["hash"] for m in self.metadata)),
            "total_chunks": len(self.metadata),
            "companies": len(self.get_companies()),
            "extractions_performed": len(self.extraction_history),
            "cache_size_mb": sum(len(pickle.dumps(v)) for v in self.embedding_cache.values()) / 1024 / 1024
        }

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🚀 Advanced Financial RAG Bot", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Financial RAG Bot")
st.markdown("**Powered by Free HuggingFace Models** 🤗")
st.caption("Extract, analyze, and compare financial metrics from earnings reports using state-of-the-art AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    hf_token = st.text_input(
        "🤗 HuggingFace API Token (Optional)",
        type="password",
        help="Get free token at https://huggingface.co/settings/tokens"
    )
    
    st.divider()
    
    model_choice = st.selectbox(
        "🤖 Select LLM Model",
        options=list(HF_MODELS.keys()),
        index=0,
        help="Mistral-7B recommended for best results"
    )
    
    st.divider()
    
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **{model_choice}**
    
    • Free via HuggingFace API
    • No GPU required
    • Rate limits apply
    """)

# Initialize bot
if 'bot' not in st.session_state:
    st.session_state.bot = AdvancedFinancialRAGBot(hf_token if hf_token else None)

bot = st.session_state.bot

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Extract", "🔍 Compare", "📈 Analytics"])

# TAB 1: Document Upload
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
            
            report_type = st.selectbox(
                "Report Type",
                ["Earnings", "10-K", "10-Q", "Annual Report", "Other"]
            )
    
    if uploaded_file and company_name:
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                result = bot.add_document(uploaded_file, company_name, report_type.lower())
            
            if result["success"]:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("New Chunks", result["chunks_added"])
                col2.metric("Total Chunks", result["total_chunks"])
                col3.metric("Duplicates", result["duplicates_skipped"])
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Show indexed companies
    if bot.metadata:
        st.divider()
        st.subheader("📚 Indexed Companies")
        companies = bot.get_companies()
        st.write(f"**{len(companies)} companies** in database:")
        st.write(", ".join(companies))

# TAB 2: Extract Financials
with tab2:
    st.header("📊 Extract Financial Metrics")
    
    companies = bot.get_companies()
    
    if not companies:
        st.warning("⚠️ No documents uploaded yet. Go to the Upload tab to add financial reports.")
    else:
        selected_company = st.selectbox("Select Company", companies)
        
        if st.button("🎯 Extract Financials", type="primary"):
            result = bot.extract_financials(selected_company, model_choice)
            
            # Display results
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Status badge
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
                    # Create DataFrame
                    df = pd.DataFrame([result.metrics.to_dict()])
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"{selected_company}_financials_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # Errors and warnings
            if result.errors:
                st.error("**Validation Errors:**")
                for error in result.errors:
                    st.write(f"• {error}")
            
            if result.warnings:
                st.warning("**Warnings:**")
                for warning in result.warnings:
                    st.write(f"• {warning}")
            
            # Show source chunks
            with st.expander("📄 View Source Chunks"):
                for i, chunk in enumerate(result.chunks_used, 1):
                    st.text_area(f"Chunk {i}", chunk, height=100)

# TAB 3: Compare Companies
with tab3:
    st.header("🔍 Compare Companies")
    
    companies = bot.get_companies()
    
    if len(companies) < 2:
        st.info("ℹ️ Upload at least 2 companies to enable comparison")
    else:
        selected_companies = st.multiselect(
            "Select Companies to Compare",
            companies,
            default=companies[:min(3, len(companies))]
        )
        
        if selected_companies and st.button("📊 Compare", type="primary"):
            with st.spinner("Extracting metrics for all companies..."):
                comparison_df = bot.compare_companies(selected_companies)
            
            if not comparison_df.empty:
                # Display comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                st.subheader("📈 Visual Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "revenue" in comparison_df.columns:
                        st.bar_chart(comparison_df.set_index("company")["revenue"])
                        st.caption("Revenue (Millions USD)")
                
                with col2:
                    if "operating_margin" in comparison_df.columns:
                        st.bar_chart(comparison_df.set_index("company")["operating_margin"])
                        st.caption("Operating Margin (%)")
                
                # Download comparison
                csv = comparison_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Comparison CSV",
                    data=csv,
                    file_name=f"company_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to extract metrics for comparison")

# TAB 4: Analytics
with tab4:
    st.header("📈 System Analytics")
    
    stats = bot.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Documents", stats["total_documents"])
    col2.metric("🧩 Chunks", stats["total_chunks"])
    col3.metric("🏢 Companies", stats["companies"])
    col4.metric("🔍 Extractions", stats["extractions_performed"])
    
    st.divider()
    
    # Extraction history
    if bot.extraction_history:
        st.subheader("📊 Extraction History")
        history_df = pd.DataFrame(bot.extraction_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Success rate
        if not history_df.empty:
            success_rate = (history_df["status"] == "success").sum() / len(history_df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.divider()
    
    # System info
    st.subheader("🔧 System Information")
    st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    st.write(f"**Cache Size:** {stats['cache_size_mb']:.2f} MB")
    st.write(f"**Index Type:** FAISS IndexFlatL2")
    
    # Clear cache option
    st.divider()
    st.subheader("⚠️ Danger Zone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all documents and extractions"):
                try:
                    # Remove all files
                    for file_path in [INDEX_PATH, METADATA_PATH, EMBEDDING_CACHE_PATH, EXTRACTION_HISTORY_PATH]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    
                    # Reset session state
                    st.session_state.bot = AdvancedFinancialRAGBot(hf_token if hf_token else None)
                    st.success("✅ All data cleared successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {e}")
    
    with col2:
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

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Advanced Financial RAG Bot</strong> | Powered by HuggingFace 🤗</p>
    <p style='font-size: 0.9em;'>
        Built with Streamlit • FAISS • Sentence Transformers • Free LLMs<br>
        <em>No API costs • Privacy-focused • Open source components</em>
    </p>
</div>
""", unsafe_allow_html=True)