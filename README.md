# FinRAG: A Retrieval-Augmented Generation System for Stock Market Insights and Investment Strategies

This repository contains **FinRAG**, a Retrieval-Augmented Generation (RAG) system designed to provide **stock market insights and investment strategy explanations** using a combination of:

- Community discussions (Reddit finance & investing subreddits),
- Historical stock CSV data,
- Finance/economics books and research papers in PDF,
- Modern embedding models and LLMs.

>  **Disclaimer:**  
> This project is for **educational and research purposes only**. It does *not* provide personalized financial advice or real-time trading recommendations.

---

## 1. Project Overview

**Goal:**  
FinRAG shows how to build an **end-to-end RAG pipeline** for finance:

1. **Collect & preprocess** financial knowledge sources (Reddit, CSV stock data, finance PDFs).
2. **Build a dense vector index** using a modern embedding model (`BAAI/bge-large-en-v1.5`) and FAISS.
3. **Retrieve relevant context** for a user’s question.
4. **Feed the retrieved context into an LLM** (e.g., Mistral-7B) to generate grounded answers.
5. **Evaluate** the system using a set of benchmark questions and semantic similarity metrics.

The system is targeted at:

- Students interested in **NLP + finance**,
- Researchers exploring **RAG architectures**,
- Anyone curious about combining **discussion data + documents + LLMs** in a structured way.

---

FinRAG: A Retrieval-Augmented Generation System for Stock Market Insights and Investment Strategies

FinRAG is a complete Retrieval-Augmented Generation (RAG) pipeline designed to generate finance-grounded explanations, stock market insights, and investment strategy analysis.

The system integrates:

Reddit financial discussions

Historical stock CSV data

Finance books & research papers (PDFs)

Modern embedding models + FAISS vector search

LLMs such as Mistral, Phi-3, and LLaMA

FinRAG demonstrates how retrieval-based LLMs can support education, research, and financial understanding — while explicitly avoiding personal financial advice.

📁 Repository Structure
FinRAG-Stock-RAG/
├── FinRAG_RAG_Pipeline.ipynb       # Full RAG pipeline notebook
├── FinRAG_Presentation.pptx        # Project presentation slides
├── README.md                       # Documentation manual (this file)
└── requirements.txt                # List of dependencies

Example dataset folder structure (optional):
dataset/
  Reddit/*.jsonl
  top10Stocks/*.csv
  books/*.pdf
  papers/*.pdf
index_bge_large/
  faiss_index.bin
  metadata.jsonl

🧠 1. Project Overview

FinRAG is built to answer finance questions by grounding LLM responses in retrieved evidence.
The pipeline performs:

✔ Data Collection & Preprocessing

Reddit posts and comments from finance-related subreddits

Stock CSV datasets

PDF books and academic papers

Automated cleaning, normalization, and chunking using LlamaParse + text splitters

✔ Vector Embedding & Indexing

Embeddings: BAAI/bge-large-en-v1.5

Vector index: FAISS

Stored metadata to ensure traceability

✔ LLM Generation

Uses HuggingFace models such as:

Mistral-7B-Instruct

Phi-3-Mini-4k-Instruct

LLaMA-3.1-Instruct

Retrieved context is injected into a structured prompt to ensure grounded answers.

✔ Evaluation

Model responses benchmarked using semantic similarity

Uses sentence-transformers/all-MiniLM-L6-v2 for lightweight automatic scoring

⚙️ 2. Installation
Create a virtual environment (recommended)
macOS / Linux:
python -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

📦 3. Requirements (requirements.txt)
torch
transformers
accelerate
sentence-transformers
faiss-cpu
pymupdf
pandas
numpy
tqdm
langchain-text-splitters
llama-parse
python-dotenv


Optional:

google-colab
scikit-learn

🚀 4. Quick Start for Students & Researchers

FinRAG can be run in Google Colab or locally.
Colab is strongly recommended for ease of use and GPU access.

⭐ Option 1 — Run in Google Colab (Recommended)

Upload the notebook:

FinRAG_RAG_Pipeline.ipynb

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


Update dataset paths:

/content/drive/MyDrive/NLP/dataset/Reddit/
/content/drive/MyDrive/NLP/dataset/books/
/content/drive/MyDrive/NLP/dataset/papers/
/content/drive/MyDrive/NLP/dataset/top10Stocks/


Run the notebook top to bottom:

Install libraries

Load datasets

Build corpus

Build FAISS index

Load LLM

Run queries

Start interactive chat:

interactive_rag()


✔ No installation
✔ GPU provided
✔ Classroom-friendly

⭐ Option 2 — Run Locally (Advanced)

Clone repository:

git clone https://github.com/<your-username>/FinRAG-Stock-RAG.git
cd FinRAG-Stock-RAG


Install dependencies:

pip install -r requirements.txt


Open notebook in Jupyter or VS Code and adjust dataset paths.

⚠️ Large models like Mistral-7B require a GPU.

Use Phi-3 Mini for CPU-friendly inference.

💬 5. Using FinRAG

After loading the model and building the index, run:

interactive_rag()


Ask questions such as:

“How does diversification reduce portfolio risk?”

“What are long-term risks of investing heavily in tech stocks?”

“What drives stock market volatility?”

FinRAG will:

Retrieve relevant Reddit posts, book excerpts, PDFs, or historical stock data

Build a grounded RAG prompt

Generate a finance-aware, evidence-backed answer

📈 6. Evaluation

FinRAG includes:

Predefined evaluation questions

Multi-model comparison (Mistral, Phi, LLaMA)

Scores computed using semantic similarity

JSONL logs for reproducibility

This helps measure:

Answer relevance

Grounding in source data

Model consistency

🛠️ 7. Extending the Project

You can extend FinRAG by:

✔ Adding More Data Sources

SEC 10-K / 10-Q filings

Financial news

Macro-economic time series

✔ Adding a Web UI

Streamlit

Gradio

✔ Adding Advanced Evaluation

Hallucination detection

Financial fact verification

📝 8. Citation

If you use FinRAG in a class project or research:

FinRAG: A Retrieval-Augmented Generation System for Stock Market Insights and Investment Strategies, University of New Haven (2025).
