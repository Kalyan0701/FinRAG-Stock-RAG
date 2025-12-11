# FinRAG: A Retrieval-Augmented Generation System for Stock Market Insights and Investment Strategies

FinRAG is a complete Retrieval-Augmented Generation (RAG) system designed to produce grounded explanations, stock market insights, and investment strategy reasoning. It retrieves information from multiple financial sources and generates context-aware responses using modern Large Language Models (LLMs).

FinRAG integrates:
- Reddit financial discussions  
- Historical stock CSV datasets  
- Finance books and research papers  
- PDF parsing with LlamaParse  
- Embedding models with FAISS vector search  
- LLMs such as Mistral, Phi-3, and LLaMA  

---

## Repository Structure
```
FinRAG-Stock-RAG/
├── FinRAG_RAG_Pipeline.ipynb # Full RAG pipeline notebook
├── FinRAG_Presentation.pptx # Project slides
├── README.md # Documentation (this file)
└── requirements.txt # Dependencies
```

Example dataset structure:
```
dataset/
Reddit/.jsonl
top10Stocks/.csv
books/.pdf
papers/.pdf
index_bge_large/
faiss_index.bin
metadata.jsonl
```

---

## 1. Project Overview

FinRAG answers finance-related questions using retrieval followed by LLM generation.

### Core Components
- Extraction and preprocessing of Reddit posts  
- Loading and cleaning stock CSV data  
- Parsing PDF books and papers  
- Chunking text and generating embeddings  
- Building a FAISS vector index  
- Retrieving relevant text chunks  
- Generating grounded answers using LLMs  

### Supported Models
- Mistral-7B-Instruct  
- Phi-3-Mini-Instruct  
- LLaMA 3.1 Instruct  

### Evaluation
- Semantic similarity scoring  
- JSONL logs for reproducibility  
- Multi-model comparison  

---

## 2. Installation

### Create a virtual environment

macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```
Windows:
```
python -m venv venv
venv\Scripts\activate
```
Install dependencies
```
pip install -r requirements.txt
```

3. Requirements
```
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
```

Optional:
```
google-colab
scikit-learn
```
4. Quick Start Guide

FinRAG can be run in Google Colab or on a local machine.

Option 1 — Run in Google Colab

Upload the notebook:
```
FinRAG_RAG_Pipeline.ipynb
```

Mount Google Drive:
```
from google.colab import drive
drive.mount('/content/drive')
```

Adjust dataset paths such as:
```
/content/drive/MyDrive/NLP/dataset/Reddit/
/content/drive/MyDrive/NLP/dataset/books/
/content/drive/MyDrive/NLP/dataset/papers/
/content/drive/MyDrive/NLP/dataset/top10Stocks/
```

Run all sections of the notebook:
```
Install libraries
Load datasets
Build the corpus
Build FAISS index
Load the LLM
Run RAG queries
Start the chat interface:
interactive_rag()
```

Option 2 — Run Locally

Clone the repository:
```
git clone https://github.com/<your-username>/FinRAG-Stock-RAG.git
cd FinRAG-Stock-RAG
```

Install dependencies:
```
pip install -r requirements.txt
```

Open the notebook and update dataset paths accordingly.
```
Note: Running large models like Mistral-7B locally requires a GPU. Use Phi-3 Mini for CPU-based inference.
```

5. Using FinRAG

After loading the index and LLM, start the interactive mode:
```
interactive_rag()
```

You can then ask questions such as:
```
How does diversification reduce portfolio risk
Why is overexposure to tech stocks risky
What factors influence long-term stock returns
FinRAG will retrieve context, construct a grounded prompt, and generate an evidence-based answer.
```
6. Evaluation Framework

FinRAG includes:
```
An evaluation dataset
Automatic semantic similarity scoring
Logs for Mistral, Phi, and LLaMA
JSONL outputs for result analysis
```

Researchers can examine:
```
Relevance
Correctness
Model differences
Stability across runs
```

7. Extending the Project

Possible extensions include:
```
Additional Data Sources
SEC filings (10-K, 10-Q)
Financial news articles
Macroeconomic datasets
Interfaces
Streamlit user interface
Gradio chatbot application
Improved Evaluation
Fact verification modules
Hallucination detection
Domain-specific benchmarking
```

8. Citation

If you use or reference this project in academic work:
```
FinRAG: A Retrieval-Augmented Generation System for Stock Market Insights and Investment Strategies, University of New Haven (2025).
```
