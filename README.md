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
