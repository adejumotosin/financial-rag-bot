import streamlit as st
import numpy as np
import faiss
import pickle
import os
from io import BytesIO
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import re
import google.generativeai as genai

# ==== CONFIG ====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "financial_index/faiss_index.bin"
METADATA_PATH = "financial_index/metadata.pkl"

# ==== INIT ====
st.set_page_config(page_title="📊 Financial Report RAG Bot", page_icon="📉", layout="wide")
os.makedirs("financial_index", exist_ok=True)

# Load FAISS + metadata
index = faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else faiss.IndexFlatL2(384)
metadata = pickle.load(open(METADATA_PATH, "rb")) if os.path.exists(METADATA_PATH) else []
model = SentenceTransformer(EMBEDDING_MODEL)

# Gemini init
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# ==== UTILS ====
def retrieve(query, top_k=5):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]


def format_financial_text(text: str) -> str:
    """Clean and format Gemini output for financial readability."""
    cleaned = (
        text.replace("\n", " ")
        .replace("  ", " ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .strip()
    )

    # Fix weird concatenations like "12,535million" -> "12,535 million"
    cleaned = re.sub(r'(\d)(million|billion|thousand)', r'\1 \2', cleaned, flags=re.IGNORECASE)

    # Bold numbers, percentages, dollar values
    cleaned = re.sub(r'(\$?\d+(?:,\d{3})*(?:\.\d+)?%?)', r'**\1**', cleaned)

    # Split into sentences for bullets if financial terms present
    sentences = [s.strip() for s in cleaned.split(".") if s.strip()]
    if any(word in cleaned.lower() for word in ["revenue", "income", "growth", "margin", "eps", "%", "$"]):
        formatted = "\n".join([f"- {s}." for s in sentences])
        return formatted

    return cleaned


def generate_with_gemini(prompt: str, max_tokens: int = 500) -> str:
    """Generate text with Gemini, format and clean it."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": max_tokens},
        )
        return format_financial_text(response.text)
    except Exception:
        # fallback to Pro if Flash quota exceeded
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": max_tokens},
            )
            return format_financial_text(response.text)
        except Exception as e2:
            return f"❌ Gemini fallback (Pro) failed: {e2}"


def process_pdf(file, company="unknown"):
    text = extract_text(file)
    sents = text.split(". ")
    chunks = [". ".join(sents[i:i + 5]) for i in range(0, len(sents), 5)]
    vectors = model.encode(chunks)
    for i, chunk in enumerate(chunks):
        index.add(np.array([vectors[i]]).astype("float32"))
        metadata.append({"id": len(metadata), "content": chunk, "company": company})
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    return len(chunks)


# ==== SESSION STATE ====
if "history" not in st.session_state:
    st.session_state.history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ==== SIDEBAR ====
st.sidebar.title("⚙️ Financial Docs Upload")
uploaded = st.sidebar.file_uploader("📂 Upload Financial PDF", type="pdf")
company_tag = st.sidebar.text_input("🏷️ Company Name (e.g., CocaCola)")

if uploaded:
    with st.spinner("🔄 Processing PDF..."):
        added = process_pdf(BytesIO(uploaded.read()), company=company_tag or "unknown")
        st.sidebar.success(f"✅ Added {added} text chunks.")

if st.sidebar.button("🧹 Reset Financial Index"):
    if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
    if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
    index.reset()
    metadata.clear()
    st.sidebar.success("Index and metadata cleared.")


# ==== MAIN CHAT ====
st.title("📊 Financial Report RAG Bot")
st.markdown("Upload financial PDFs and ask questions. Gemini will summarize with context.")

st.metric("📄 Documents", len(set([m["company"] for m in metadata])))
st.metric("🧩 Chunks", len(metadata))
st.metric("💬 Questions Asked", st.session_state.question_count)
if metadata:
    st.success(f"✅ Ready with {len(metadata)} chunks from {len(set([m['company'] for m in metadata]))} companies!")


query = st.text_input("💬 Ask your question:")
if query:
    with st.spinner("Thinking..."):
        chunks = retrieve(query)
        context = "\n\n".join(chunk["content"] for chunk in chunks)

        prompt = f"""You are a financial assistant analyzing company financial documents. 
Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

        answer = generate_with_gemini(prompt)
        st.session_state.history.append((query, answer))
        st.session_state.question_count += 1

# ==== DISPLAY CHAT HISTORY ====
if st.session_state.history:
    st.markdown("### 🗣️ Chat History")
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(a)
        st.markdown("---")

# ==== SOURCES ====
if query:
    with st.expander("📄 Top Matching Context Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}** *(Company: {chunk.get('company', 'N/A')})*:\n{chunk['content']}")