import streamlit as st
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re
import json
from datetime import datetime
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

# =========================
# CONFIG
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"
EMBEDDING_CACHE_PATH = "financial_index/embedding_cache.pkl"
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]
CHUNK_SIZE = 1000
OVERLAP = 200

# =========================
# FINANCIAL REPORT BOT
# =========================
class FinancialRAGBot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.metadata = []
        self.embedding_cache = {}
        self.index = None

        # --- FIX: Create the directory if it doesn't exist ---
        os.makedirs("financial_index", exist_ok=True)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)

        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                self.embedding_cache = pickle.load(f)

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)

    def embed_text(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        emb = self.model.encode([text])[0].astype("float32")
        self.embedding_cache[text] = emb
        return emb

    def save(self):
        if self.index:
            faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def add_document(self, file, company: str):
        text = extract_text(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
        chunks = splitter.split_text(text)

        vectors = []
        new_metadata = []

        for chunk in chunks:
            emb = self.embed_text(chunk)
            vectors.append(emb)
            new_metadata.append({
                "company": company,
                "content": chunk,
                "source": file.name,
                "hash": hashlib.md5(chunk.encode()).hexdigest()
            })

        vectors = np.vstack(vectors)

        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.metadata.extend(new_metadata)
        self.save()

    def generate_with_gemini(self, prompt: str, max_tokens=500):
        """Try Gemini models in fallback order."""
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
                if response and response.candidates:
                    # Sometimes .text is empty, so fallback to parts
                    text = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
                    return text, model_name
            except Exception:
                continue
        return "", None

    def extract_financials(self, company: str):
        """Extract financial metrics into CSV-ready DataFrame."""
        chunks = [m for m in self.metadata if m["company"] == company]
        if not chunks:
            return pd.DataFrame()

        all_data = []
        debug_logs = []

        for start in range(0, len(chunks), 20):
            batch = chunks[start:start+20]
            context = "\n\n".join(c["content"] for c in batch)

            prompt = f"""
Extract financial metrics from the following report text and return ONLY valid JSON.
Do not include explanations or markdown.

Expected JSON schema:
[
  {{
    "Company": "{company}",
    "Quarter": "QX YYYY",
    "Revenue": number,
    "OperatingIncome": number,
    "OperatingMargin": number,
    "NetIncome": number,
    "EPS": number,
    "ComparableEPS": number
  }}
]

TEXT:
{context}
"""

            text, model_used = self.generate_with_gemini(prompt, max_tokens=800)
            debug_logs.append((start, text))

            try:
                cleaned = re.sub(r"```json|```", "", text).strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    all_data.extend(parsed)
            except Exception:
                continue

        if all_data:
            df = pd.DataFrame(all_data).drop_duplicates()
            return df
        else:
            st.warning("⚠️ No structured data parsed. Check debug logs below.")
            with st.expander("🐞 Debug: Raw Gemini Outputs"):
                for batch_id, raw in debug_logs:
                    st.text_area(f"Batch {batch_id}", raw, height=200)
            return pd.DataFrame()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="📊 Financial Report RAG Bot", layout="wide")

st.title("📊 Financial Report RAG Bot")
st.caption("Upload financial PDFs and ask questions. Gemini will summarize with context and source citation.")

api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password")

if api_key:
    bot = FinancialRAGBot(api_key)

    st.subheader("⚙️ Document Management")
    uploaded = st.file_uploader("📂 Upload Financial PDF", type=["pdf"])

    if uploaded:
        company_name = st.text_input("🏷️ Company Name", value=uploaded.name.split(".")[0])
        if st.button("Process Document"):
            bot.add_document(uploaded, company_name)
            st.success(f"✅ Processed {uploaded.name} for {company_name}")

    if bot.metadata:
        st.subheader("📊 Financial Report RAG Bot")
        company_for_extract = st.selectbox("Select company to extract financials:", sorted(set(m["company"] for m in bot.metadata)))
        
        if st.button("📤 Extract Financials to CSV"):
            df = bot.extract_financials(company_for_extract)
            if not df.empty:
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name=f"{company_for_extract}_financials.csv", mime="text/csv")
            else:
                st.error("❌ No financials could be extracted.")
