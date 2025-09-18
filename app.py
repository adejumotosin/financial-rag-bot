# =========================
# IMPORTS
# =========================
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

LLM_MODELS = [
    "claude-3-opus-20240229",   # Anthropic Claude 3 Opus
    "gemini-1.5-pro-latest"     # Google Gemini Pro
]

CHUNK_SIZE = 1000
OVERLAP = 200


# =========================
# FINANCIAL RAG BOT CLASS
# =========================
class FinancialRAGBot:
    def __init__(self, api_keys: Dict[str, str]):
        if api_keys.get("google"):
            genai.configure(api_key=api_keys["google"])
        
        self.anthropic_client = None
        if api_keys.get("anthropic"):
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=api_keys["anthropic"])

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.metadata = []
        self.embedding_cache = {}
        self.index = None

        os.makedirs("financial_index", exist_ok=True)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "rb") as f:
                self.embedding_cache = pickle.load(f)
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)

    def generate_with_llm(self, prompt: str, max_tokens=2000):
        """Try LLMs in fallback order (Claude → Gemini)."""
        for model_name in LLM_MODELS:
            try:
                if model_name.startswith("claude-") and self.anthropic_client:
                    response = self.anthropic_client.messages.create(
                        model=model_name,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text, model_name

                elif model_name.startswith("gemini-") and getattr(genai, "api_key", None):
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        prompt, 
                        generation_config={"max_output_tokens": max_tokens}
                    )
                    if response and response.candidates:
                        text = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
                        return text, model_name

            except Exception as e:
                st.warning(f"⚠️ Error with {model_name}: {e}")
                continue

        return "", None

    def add_document(self, file, company: str):
        text = extract_text(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
        chunks = splitter.split_text(text)

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())

        for chunk in chunks:
            h = hashlib.sha256(chunk.encode()).hexdigest()
            if h not in self.embedding_cache:
                embedding = self.model.encode([chunk])[0]
                self.embedding_cache[h] = embedding
                self.index.add(np.array([embedding], dtype="float32"))
                self.metadata.append({"company": company, "content": chunk, "hash": h})

        self.save()

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(EMBEDDING_CACHE_PATH, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def extract_financials(self, company: str):
        full_text = " ".join([m["content"] for m in self.metadata if m["company"] == company])
        if not full_text:
            return pd.DataFrame()

        prompt = f"""
You are an expert financial analyst. Extract key financial metrics from the following financial report text.

Return ONLY a valid JSON object (no markdown, no commentary).

Schema: {{
  "Company": "string",
  "Quarter": "string (e.g., Q1 2024, FY 2024)",
  "Revenue": number (in millions USD),
  "OperatingIncome": number (in millions USD),
  "OperatingMargin": number (percentage, e.g., 34.1),
  "NetIncome": number (in millions USD),
  "EPS": number,
  "ComparableEPS": number
}}

Rules:
1. Focus ONLY on the most recent quarter or fiscal year.
2. Use the company "{company}" only.
3. Normalize all amounts to millions USD.
   "$12.5 billion" → 12500
   "$12,500 million" → 12500
4. If a metric is missing, use null.
5. Pick the most explicit & final figure (tables > narrative).
6. Do not output explanations.

TEXT: {full_text}
"""

        text, model_used = self.generate_with_llm(prompt, max_tokens=2000)
        if not text:
            st.warning("⚠️ No text returned by LLM.")
            return pd.DataFrame()

        try:
            cleaned = re.sub(r"```json|```", "", text).strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                st.warning("⚠️ Unexpected JSON format.")
                return pd.DataFrame()

            data = parsed
            errors = []

            if data.get("Company") != company:
                errors.append("Company mismatch. Forcing correction.")
                data["Company"] = company

            required = ["Revenue", "NetIncome", "Quarter"]
            for r in required:
                if data.get(r) is None:
                    errors.append(f"Missing required: {r}")

            for key in ["Revenue", "OperatingIncome", "OperatingMargin", "NetIncome", "EPS", "ComparableEPS"]:
                val = data.get(key)
                if val is not None:
                    try:
                        data[key] = float(val)
                    except:
                        errors.append(f"{key} invalid number: {val}")
                        data[key] = None

            if errors:
                st.warning("⚠️ Validation issues found:")
                for e in errors:
                    st.write("- " + e)
                st.text_area("🐞 Debug Output", cleaned, height=300)

            return pd.DataFrame([data])

        except Exception as e:
            st.warning(f"⚠️ Failed to parse JSON: {e}")
            st.text_area("🐞 Raw Output", text, height=300)
            return pd.DataFrame()


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="📊 Financial Report RAG Bot", layout="wide")

st.title("📊 Financial Report RAG Bot")
st.caption("Upload financial PDFs and extract structured data with Claude/Gemini.")

st.sidebar.subheader("🔑 API Keys")
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic Claude API Key", type="password")

api_keys = {}
if google_api_key:
    api_keys["google"] = google_api_key
if anthropic_api_key:
    api_keys["anthropic"] = anthropic_api_key

if api_keys:
    bot = FinancialRAGBot(api_keys)

    st.subheader("⚙️ Document Management")
    uploaded = st.file_uploader("📂 Upload Financial PDF", type=["pdf"])

    if uploaded:
        company_name = st.text_input("🏷️ Company Name", value=uploaded.name.split(".")[0])
        if st.button("Process Document"):
            if not bot.anthropic_client and not getattr(genai, "api_key", None):
                st.error("Please provide at least one LLM API key.")
            else:
                bot.add_document(uploaded, company_name)
                st.success(f"✅ Processed {uploaded.name} for {company_name}")

    if bot.metadata:
        st.subheader("📊 Extract Financials")
        company_for_extract = st.selectbox("Select company to extract:", sorted(set(m["company"] for m in bot.metadata)))

        if st.button("📤 Extract Financials to CSV"):
            if not bot.anthropic_client and not getattr(genai, "api_key", None):
                st.error("Please configure an API key to extract financials.")
            else:
                df = bot.extract_financials(company_for_extract)
                if not df.empty:
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download CSV", 
                        data=csv, 
                        file_name=f"{company_for_extract}_financials.csv", 
                        mime="text/csv"
                    )
                else:
                    st.error("❌ No valid financials could be extracted.")
else:
    st.info("Please enter at least one LLM API key in the sidebar to begin.")