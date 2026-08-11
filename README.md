# Financial RAG Bot

A Streamlit-based financial research application for extracting, retrieving, validating, and comparing company financial metrics from uploaded earnings reports.

The project combines document parsing, semantic retrieval, local vector indexing, and LLM-assisted structured extraction to turn unstructured financial PDFs into analysis-ready metrics.

## What it does

- Upload earnings reports and other financial PDFs
- Extract document text with `pdfminer.six`
- Split reports into overlapping semantic chunks
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Store and search embeddings with FAISS, with a NumPy fallback index
- Retrieve company-specific financial context using semantic search
- Use Groq-hosted language models for structured financial extraction
- Extract revenue, operating income, margins, net income, EPS, assets, equity, and operating cash flow when available
- Validate dates, ranges, margins, and basic financial consistency
- Assign extraction confidence scores and success/partial/failed states
- Compare extracted metrics across multiple companies
- Persist embedding metadata, caches, and extraction history locally

## Architecture

```text
Financial PDF
    |
    v
PDF text extraction
    |
    v
Chunking + deduplication
    |
    v
MiniLM embeddings
    |
    v
FAISS / local vector index
    |
    v
Company-aware semantic retrieval
    |
    v
Groq LLM structured extraction
    |
    v
Validation + confidence scoring
    |
    v
Streamlit analysis interface
```

## Extracted financial fields

The current data model supports:

- Revenue
- Operating income
- Operating margin
- Net income
- Basic EPS
- Diluted EPS
- Gross profit
- Gross margin
- Total assets
- Total equity
- Cash flow from operations
- Reporting period and period-end date

Not every report exposes every field in a reliably retrievable form, so missing metrics are preserved as missing rather than fabricated.

## Retrieval and extraction workflow

The application uses several targeted semantic queries for each company, including income statement, revenue, operating income, EPS, and quarterly-result terminology. Retrieved chunks are deduplicated before they are passed into the extraction prompt.

The extraction layer requests JSON output, normalizes the result into a typed financial metrics object, applies validation rules, and calculates a confidence score based on field coverage and consistency checks.

## Run locally

```bash
git clone https://github.com/adejumotosin/financial-rag-bot.git
cd financial-rag-bot

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a Groq API key in the application sidebar to enable LLM extraction.

## Technology

- Python
- Streamlit
- pandas / NumPy
- Sentence Transformers
- FAISS
- LangChain text splitting
- PDFMiner
- Groq API

## Current limitations

- PDF extraction quality depends on the structure and text layer of the source report.
- Scanned or image-only PDFs require an OCR layer that is not currently integrated.
- Financial values are extracted with an LLM and should be checked against the source report before production use.
- The current persistence layer is local file storage rather than a production database.
- The application is a research tool, not an audited financial-data service.

## Roadmap

- Add source citations down to page and table level
- Add OCR for scanned filings
- Replace local persistence with a database-backed document store
- Add period-over-period financial trend analysis
- Add multi-document company timelines
- Add benchmark and peer-group comparison
- Add automated regression tests for extraction accuracy
- Add structured export to CSV and JSON

## Disclaimer

This repository is for research and educational use. Extracted financial information should be verified against the original filing before it is used for investment, accounting, or other high-stakes decisions.
