"""
app.py

Single-file application that provides:
- Streamlit UI for uploading PDFs (single & batch), extracting financial metrics with OCR fallback
- Export to CSV / Excel
- Visual dashboard & charts
- FastAPI microservice (endpoints for single/batch extract and semantic search)
- Vector DB support: Chroma or Qdrant if available, otherwise FAISS/simple fallback
- Persistent storage for metadata, embeddings, vectors, and extraction history

Run UI:
    streamlit run app.py

Run API:
    uvicorn app:api --reload --port 8000

Notes:
- Optional extras: install pytesseract, pdf2image, chromadb, qdrant-client, faiss-cpu for best behavior.
- This file uses safe fallbacks when optional libraries are not available.
"""

import os
import io
import re
import json
import time
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Streamlit UI
import streamlit as st
import pandas as pd
import numpy as np

# PDF text extraction
from pdfminer.high_level import extract_text

# OCR fallback
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    S2_AVAILABLE = False

# Vector DB options
_FAISS_AVAILABLE = True
try:
    import faiss
except Exception:
    _FAISS_AVAILABLE = False

CHROMA_AVAILABLE = False
QDRANT_AVAILABLE = False
try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# -------------------------
# Configuration & Paths
# -------------------------
DATA_DIR = "financial_index"
os.makedirs(DATA_DIR, exist_ok=True)

METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
EMBEDDING_CACHE_PATH = os.path.join(DATA_DIR, "embedding_cache.pkl")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.pkl")
EXTRACTION_HISTORY_PATH = os.path.join(DATA_DIR, "extraction_history.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fallback string
EMBEDDING_DIM_FALLBACK = 384

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# Vector DB selection: "chroma", "qdrant", "faiss", "simple"
VECTOR_DB_PREFERRED = os.environ.get("VECTOR_DB", "chroma")  # can be changed by env var

# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str, default=None):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default

# -------------------------
# Embedding model loader (cached)
# -------------------------
@st.cache_resource
def load_embedding_model(name: str = EMBEDDING_MODEL_NAME):
    if not S2_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    return SentenceTransformer(name)

def get_embedding_model():
    try:
        return load_embedding_model()
    except Exception:
        # last-resort: try to construct with short name
        if S2_AVAILABLE:
            return SentenceTransformer("all-MiniLM-L6-v2")
        raise

# -------------------------
# Text splitting utilities
# -------------------------
def simple_split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r'\n{2,}', '\n\n', text.strip())
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        start = 0
        while start < len(p):
            end = start + chunk_size
            chunks.append(p[start:end])
            start = end - overlap if end - overlap > start else end
    if not chunks and text:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
    return chunks

# -------------------------
# OCR fallback functions
# -------------------------
def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    if not OCR_AVAILABLE:
        return ""
    pages = convert_from_bytes(pdf_bytes, dpi=200)
    texts = []
    for p in pages:
        texts.append(pytesseract.image_to_string(p))
    return "\n\n".join(texts)

# -------------------------
# Simple Vector Index fallback
# -------------------------
class SimpleIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
    def add(self, arr: np.ndarray):
        arr = np.asarray(arr, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.vectors = np.vstack([self.vectors, arr])
    def search(self, query: np.ndarray, k: int = 5):
        if self.vectors.shape[0] == 0:
            return np.array([[]]), np.array([[]])
        q = np.asarray(query, dtype="float32")
        diffs = self.vectors - q
        dists = np.sum(diffs * diffs, axis=1)
        idx = np.argsort(dists)[:k]
        return np.array([dists[idx]]), np.array([idx])

# -------------------------
# Vector DB abstraction
# -------------------------
class VectorDB:
    def __init__(self, method: str = None, dim: int = EMBEDDING_DIM_FALLBACK):
        self.method = (method or VECTOR_DB_PREFERRED).lower()
        self.dim = dim
        self._client = None
        self._collection = None
        self._faiss_index = None
        self._simple_index = None
        self._ids: List[str] = []
        self._vectors_in_memory: List[np.ndarray] = []

        # Try to initialize chosen backend; fallbacks provided
        if self.method == "chroma" and CHROMA_AVAILABLE:
            try:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=DATA_DIR))
                self._client = client
                self._collection = client.get_or_create_collection(name="financial_chunks")
            except Exception:
                self.method = "faiss"
        if self.method == "qdrant" and QDRANT_AVAILABLE:
            try:
                # uses default local qdrant or env-configured
                self._client = QdrantClient()
                self._collection = None  # manage on-demand
            except Exception:
                self.method = "faiss"
        if self.method == "faiss" and _FAISS_AVAILABLE:
            self._faiss_index = faiss.IndexFlatL2(self.dim)
        if self._faiss_index is None and self.method != "chroma" and self.method != "qdrant":
            # fallback to simple index
            self._simple_index = SimpleIndex(self.dim)

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        # ids optional
        if ids is None:
            ids = [sha256(json.dumps(m) + str(time.time()) + str(i)) for i,m in enumerate(metadatas)]
        if self.method == "chroma" and self._collection is not None:
            texts = [m.get("content","") for m in metadatas]
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=[v.tolist() for v in vectors])
            return ids
        if self.method == "qdrant" and self._client is not None:
            # simple upsert to 'financial_chunks' collection
            try:
                self._client.upsert(collection_name="financial_chunks",
                                    points=[{"id": int(i if isinstance(i,int) else int(int(hashlib.sha1(i.encode()).hexdigest(),16)%1_000_000_000)),
                                             "vector": v.tolist(), "payload": m} for i,(v,m) in enumerate(zip(vectors, metadatas))])
                return ids
            except Exception:
                pass
        if self._faiss_index is not None:
            arr = np.vstack([np.asarray(v, dtype="float32") for v in vectors])
            self._faiss_index.add(arr)
            self._vectors_in_memory.extend([np.asarray(v, dtype="float32") for v in vectors])
            self._ids.extend(ids)
            return ids
        # fallback simple
        for v in vectors:
            self._simple_index.add(np.asarray(v, dtype="float32"))
            self._vectors_in_memory.append(np.asarray(v, dtype="float32"))
        self._ids.extend(ids)
        return ids

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.asarray(query_vector, dtype="float32")
        if self.method == "chroma" and self._collection is not None:
            results = self._collection.query(q.tolist(), n_results=top_k, include=["metadatas","distances"])
            hits = []
            for i, d in enumerate(results["distances"]):
                for j,dist in enumerate(d):
                    md = results["metadatas"][i][j]
                    hits.append((md, float(dist)))
            return hits
        if self.method == "qdrant" and self._client is not None:
            try:
                resp = self._client.search(collection_name="financial_chunks", query_vector=q.tolist(), limit=top_k)
                return [(hit.payload, hit.score) for hit in resp]
            except Exception:
                pass
        if self._faiss_index is not None:
            D, I = self._faiss_index.search(np.array([q], dtype="float32"), top_k)
            hits = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                md = {"id": self._ids[idx], "content": (self._vectors_in_memory[idx] if False else None)}
                hits.append((md, float(dist)))
            return hits
        # simple index search
        D, I = self._simple_index.search(q, top_k)
        hits = []
        if D.size == 0:
            return []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._vectors_in_memory):
                continue
            hits.append(({"id": self._ids[idx]}, float(dist)))
        return hits

    def persist(self):
        # persist simple vectors for fallback
        try:
            if self._vectors_in_memory:
                save_pickle(self._vectors_in_memory, VECTORS_PATH)
                save_pickle(self._ids, os.path.join(DATA_DIR, "vector_ids.pkl"))
        except Exception:
            pass

    def load_persisted(self):
        vecs = load_pickle(VECTORS_PATH, default=[])
        ids = load_pickle(os.path.join(DATA_DIR, "vector_ids.pkl"), default=[])
        if vecs and self._faiss_index is not None:
            arr = np.vstack([np.asarray(v, dtype="float32") for v in vecs])
            self._faiss_index.add(arr)
            self._vectors_in_memory = [np.asarray(v, dtype="float32") for v in vecs]
            self._ids = ids
        elif vecs and self._simple_index is not None:
            for v in vecs:
                self._simple_index.add(np.asarray(v, dtype="float32"))
            self._vectors_in_memory = [np.asarray(v, dtype="float32") for v in vecs]
            self._ids = ids

# -------------------------
# Core extraction & RAG logic
# -------------------------
class FinancialExtractor:
    def __init__(self, embedding_model=None, vector_db: VectorDB = None):
        self.embedding_model = embedding_model or (get_embedding_model() if S2_AVAILABLE else None)
        self.dim = getattr(self.embedding_model, "get_sentence_embedding_dimension", lambda: EMBEDDING_DIM_FALLBACK)()
        self.vector_db = vector_db or VectorDB(method=VECTOR_DB_PREFERRED, dim=self.dim)
        self.metadata = load_pickle(METADATA_PATH, default=[])
        self.embedding_cache = load_pickle(EMBEDDING_CACHE_PATH, default={})
        self.extraction_history = load_pickle(EXTRACTION_HISTORY_PATH, default=[])
        # load persisted vectors if any
        try:
            self.vector_db.load_persisted()
        except Exception:
            pass

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not available.")
        embs = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embs, dtype="float32")

    def ingest_document(self, file_bytes: bytes, company: str, report_type: str = "earnings") -> Dict[str, Any]:
        # extract text (pdfminer), fallback to OCR if needed
        text = extract_text(io.BytesIO(file_bytes))
        used_ocr = False
        if not text or len(text.strip()) < 50:
            if OCR_AVAILABLE:
                text = ocr_pdf_bytes(file_bytes)
                used_ocr = True
            else:
                return {"success": False, "error": "Document has no extractable text and OCR not available."}

        chunks = simple_split_text(text)
        # create embeddings for chunks not seen before (via hash)
        new_chunks = []
        new_embs = []
        new_meta = []
        for c in chunks:
            h = sha256(c)
            if h in self.embedding_cache:
                continue
            new_chunks.append(c)
            new_meta.append({"company": company, "content": c, "hash": h, "report_type": report_type, "added_date": now_iso()})
        if new_chunks:
            embs = self._embed(new_chunks)
            # add to vector DB
            ids = [m["hash"] for m in new_meta]
            self.vector_db.add([e for e in embs], new_meta, ids)
            # cache embeddings
            for m,e in zip(new_meta, embs):
                self.embedding_cache[m["hash"]] = e
            # extend metadata
            self.metadata.extend(new_meta)
            self._persist_state()
            self.vector_db.persist()
        return {"success": True, "chunks_added": len(new_meta), "total_chunks": len(chunks), "used_ocr": used_ocr}

    def _persist_state(self):
        save_pickle(self.metadata, METADATA_PATH)
        save_pickle(self.embedding_cache, EMBEDDING_CACHE_PATH)
        save_pickle(self.extraction_history, EXTRACTION_HISTORY_PATH)

    def semantic_search(self, query: str, company: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._embed([query])[0]
        hits = self.vector_db.search(q_emb, top_k)
        results = []
        for payload, dist in hits:
            meta = payload if isinstance(payload, dict) else {}
            if company and meta.get("company") != company:
                continue
            results.append({"content": meta.get("content"), "company": meta.get("company"), "distance": dist})
        return results

    def extract_financials_from_company(self, company: str, top_k: int = 10) -> Dict[str, Any]:
        # Query for common financial terms and then attempt to parse metrics via regex heuristics
        search_query = f"{company} consolidated statements revenue net income operating income eps earnings per share"
        hits = self.semantic_search(search_query, company=company, top_k=top_k)
        if not hits:
            return {"success": False, "error": "No chunks found for company."}
        combined_text = "\n\n".join([h["content"] for h in hits])
        parsed = self.parse_financials_heuristic(combined_text)
        confidence = self.calculate_confidence(parsed, [h["content"] for h in hits])
        result = {"company": company, "parsed": parsed, "confidence": confidence, "chunks_used": [h["content"] for h in hits]}
        # store extraction history
        self.extraction_history.append({"company": company, "timestamp": now_iso(), "result": result})
        self._persist_state()
        return {"success": True, "result": result}

    def parse_financials_heuristic(self, text: str) -> Dict[str, Optional[float]]:
        # Very pragmatic regex-based extraction for common items. Returns None when not found.
        def find_money(labels: List[str]) -> Optional[float]:
            patterns = []
            for lab in labels:
                patterns.append(rf"{lab}[:\s]*\$?\s*([0-9\.,]+)\s*(M|B|bn|m|b)?")
                patterns.append(rf"{lab}.*?\n.*?([0-9\.,]+)\s*(M|B|bn|m|b)?")
            for p in patterns:
                m = re.search(p, text, re.IGNORECASE)
                if m:
                    num = m.group(1).replace(",", "")
                    scale = (m.group(2) or "").lower()
                    try:
                        val = float(num)
                        if scale in ("m", "m", "mn"):
                            return val
                        if scale in ("b", "bn"):
                            return val * 1000.0
                        # if no scale assume units are millions or raw; to be safe return raw
                        return val
                    except Exception:
                        continue
            return None

        # fields and synonyms
        result = {}
        result["revenue"] = find_money(["revenue", "total revenue", "net sales", "sales"])
        result["operating_income"] = find_money(["operating income", "operating profit", "income from operations"])
        result["net_income"] = find_money(["net income", "net earnings", "profit after tax", "profit for the period"])
        result["eps"] = find_money(["earnings per share", "basic earnings per share", "diluted earnings per share"])
        result["gross_profit"] = find_money(["gross profit"])
        # Attempt to find margins
        gm = re.search(r"(gross margin|gross profit margin)[:\s]*([0-9]{1,3}\.?\d?)\s*%", text, re.IGNORECASE)
        if gm:
            result["gross_margin"] = float(gm.group(2))
        om = re.search(r"(operating margin|operating profit margin)[:\s]*([0-9]{1,3}\.?\d?)\s*%", text, re.IGNORECASE)
        if om:
            result["operating_margin"] = float(om.group(2))
        return result

    def calculate_confidence(self, parsed: Dict[str, Any], chunks: List[str]) -> float:
        score = 0.0
        if parsed.get("revenue") and parsed.get("net_income"):
            score += 40
        if parsed.get("operating_income"):
            score += 20
        if parsed.get("eps"):
            score += 10
        if parsed.get("gross_profit") or parsed.get("gross_margin"):
            score += 10
        # presence of "consolidated statements" in chunks
        if any("consolidated" in c.lower() for c in chunks):
            score += 20
        return min(score, 100.0)

    def batch_ingest(self, files: List[bytes], company_names: List[str], report_types: List[str]) -> List[Dict[str, Any]]:
        results = []
        for fb, comp, rtype in zip(files, company_names, report_types):
            res = self.ingest_document(fb, comp, rtype)
            results.append({"company": comp, "result": res})
        return results

# -------------------------
# Instantiate core components
# -------------------------
EMBED_MODEL = None
try:
    EMBED_MODEL = get_embedding_model() if S2_AVAILABLE else None
except Exception:
    EMBED_MODEL = None

VECTOR_DB = VectorDB(method=VECTOR_DB_PREFERRED, dim=getattr(EMBED_MODEL, "get_sentence_embedding_dimension", lambda: EMBEDDING_DIM_FALLBACK)())
EXTRACTOR = FinancialExtractor(embedding_model=EMBED_MODEL, vector_db=VECTOR_DB)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Financial Extractor + API", layout="wide")
st.title("Automated Financial Extractor — Streamlit + FastAPI")
st.markdown("Upload single or multiple PDFs, extract financial metrics with OCR fallback, store contextual chunks in a vector DB, visualize results, and export to CSV/Excel.")

with st.sidebar:
    st.header("System")
    st.write(f"Vector DB: {VECTOR_DB.method}")
    st.write(f"OCR available: {OCR_AVAILABLE}")
    st.write(f"Sentence-Transformers available: {S2_AVAILABLE}")
    if st.button("Reload vector DB from disk"):
        VECTOR_DB.load_persisted()
        st.experimental_rerun()

tab_upload, tab_extract, tab_search, tab_history = st.tabs(["Upload & Ingest", "Extract & Dashboard", "Semantic Search", "History & Export"])

with tab_upload:
    st.header("Upload PDFs (single or batch)")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        cols = st.columns(3)
        company_names = []
        report_types = []
        for i, f in enumerate(uploaded_files):
            with cols[i % 3]:
                name = st.text_input(f"Company name for {f.name}", value=f.name.rsplit(".",1)[0].replace("_"," ").title(), key=f"name_{i}")
                company_names.append(name)
                rtype = st.selectbox(f"Report type for {f.name}", ["Earnings","10-K","10-Q","Annual Report","Other"], key=f"type_{i}")
                report_types.append(rtype.lower())
        if st.button("Ingest uploaded PDFs"):
            pb = [f.read() for f in uploaded_files]
            with st.spinner("Ingesting and embedding..."):
                results = EXTRACTOR.batch_ingest(pb, company_names, report_types)
            st.success("Ingestion completed")
            for r in results:
                st.write(r)

with tab_extract:
    st.header("Extract financials for a company (heuristic)")
    companies = sorted({m.get("company") for m in EXTRACTOR.metadata if m.get("company")})
    companies = [c for c in companies if c]
    if not companies:
        st.info("No companies indexed yet. Upload PDFs in 'Upload & Ingest'")
    else:
        selected = st.selectbox("Select company", companies)
        k = st.slider("Chunks to consider (top-K)", 1, 20, 8)
        if st.button("Run extraction"):
            with st.spinner("Running semantic search and heuristic parsing..."):
                res = EXTRACTOR.extract_financials_from_company(selected, top_k=k)
            if res.get("success"):
                r = res["result"]
                st.metric("Confidence", f"{r['confidence']:.0f}%")
                parsed = r["parsed"]
                df = pd.DataFrame([parsed])
                st.subheader("Parsed Financial Metrics (heuristic)")
                st.dataframe(df)
                # Charts
                numeric_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x,(int,float))).any()]
                if numeric_cols:
                    st.subheader("Charts")
                    chart_df = df[numeric_cols].T
                    chart_df.columns = ["value"]
                    st.bar_chart(chart_df)
                # Export
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv_bytes, file_name=f"{selected}_metrics_{datetime.utcnow().date()}.csv", mime="text/csv")
                # Excel
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="metrics")
                st.download_button("Download Excel", towrite.getvalue(), file_name=f"{selected}_metrics.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.error(res.get("error","Extraction failed"))

with tab_search:
    st.header("Semantic search over indexed chunks")
    q = st.text_input("Search query")
    comp_filter = st.text_input("Company filter (optional)")
    top_k = st.slider("Top K", 1, 20, 5)
    if st.button("Search"):
        if not q:
            st.warning("Provide a query")
        else:
            hits = EXTRACTOR.semantic_search(q, company=comp_filter if comp_filter else None, top_k=top_k)
            st.write(f"{len(hits)} hits")
            for i,h in enumerate(hits,1):
                st.write(f"--- Hit {i} (distance {h['distance']:.3f}) ---")
                st.write(h["content"][:1000])

with tab_history:
    st.header("Extraction History & Persistence")
    hist = EXTRACTOR.extraction_history
    if hist:
        dfh = pd.DataFrame(hist)
        st.dataframe(dfh)
        csvb = dfh.to_csv(index=False).encode("utf-8")
        st.download_button("Download Extraction History CSV", csvb, "extraction_history.csv")
    else:
        st.info("No extraction history yet.")

# -------------------------
# FastAPI app (microservice)
# -------------------------
api = FastAPI(title="Financial Extraction API", version="1.0")

@api.get("/health")
async def health():
    return JSONResponse({"status": "ok", "time": now_iso(), "vector_db": VECTOR_DB.method, "ocr": OCR_AVAILABLE, "s2": S2_AVAILABLE})

@api.post("/extract/single")
async def api_extract_single(file: UploadFile = File(...), company: str = Form(...), report_type: str = Form("earnings")):
    try:
        fb = await file.read()
        res = EXTRACTOR.ingest_document(fb, company, report_type)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error"))
        # perform an immediate extraction attempt
        extract_res = EXTRACTOR.extract_financials_from_company(company, top_k=10)
        return JSONResponse({"ingest": res, "extraction": extract_res})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/extract/batch")
async def api_extract_batch(files: List[UploadFile] = File(...), company_names: List[str] = Form(...), report_types: List[str] = Form(...)):
    # company_names, report_types expected as JSON arrays (strings) from client
    try:
        # convert to lists if sent as JSON strings
        if isinstance(company_names, str):
            company_names = json.loads(company_names)
        if isinstance(report_types, str):
            report_types = json.loads(report_types)
    except Exception:
        pass
    if len(files) != len(company_names) or len(files) != len(report_types):
        raise HTTPException(status_code=400, detail="files, company_names and report_types must be same length")
    fb_list = [await f.read() for f in files]
    res = EXTRACTOR.batch_ingest(fb_list, company_names, report_types)
    return JSONResponse({"results": res})

@api.post("/search")
async def api_search(query: str = Form(...), company: Optional[str] = Form(None), top_k: int = Form(5)):
    try:
        hits = EXTRACTOR.semantic_search(query, company=company, top_k=top_k)
        return JSONResponse({"query": query, "hits": hits})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: run API if executed as module with "api" arg
if __name__ == "__main__":
    import sys
    mode = "streamlit"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "api":
        mode = "api"
    if mode == "api":
        uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)
    else:
        # Launch Streamlit UI (this will run when streamlit executes the file)
        # The Streamlit app code above runs as part of import; nothing extra needed here.
        pass
