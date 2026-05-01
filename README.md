# RAG From Scratch — Episode 2

<img width="700" height="197" alt="image" src="https://github.com/user-attachments/assets/8160d49f-450e-441c-8e45-ccb90d284666" />




# RAG From Scratch — Episode 2

## Overview

This project demonstrates advanced Retrieval-Augmented Generation (RAG) techniques using LangChain, OpenAI models, and vector databases.

The notebook explores multiple retrieval strategies beyond basic RAG, including:

* Multi-Query Retrieval
* RAG Fusion
* Query Decomposition
* Reciprocal Rank Fusion (RRF)
* Context-aware generation pipelines

The goal is to improve retrieval quality, reduce hallucinations, and increase answer accuracy by generating better search queries and combining retrieval results intelligently.

---

# Features

## 1. Multi-Query Retrieval

Instead of relying on a single user query, the system generates multiple semantically related search queries.

### Benefits

* Improves document recall
* Captures different perspectives of the same question
* Reduces missed context during retrieval

### Example

User Question:

```text
What are the impacts of AI on software engineering?
```

Generated Queries:

```text
- How is AI changing software development?
- Effects of AI in software engineering workflows
- AI tools used in software engineering
- Future of AI in programming
```

---

## 2. RAG Fusion

RAG Fusion combines results from multiple retrieval queries and reranks them using Reciprocal Rank Fusion (RRF).

### Benefits

* Reduces noisy retrievals
* Boosts documents appearing across multiple searches
* Produces more reliable context

---

## 3. Query Decomposition

Complex questions are broken into smaller sub-questions.

### Benefits

* Handles multi-hop reasoning better
* Improves retrieval for complicated prompts
* Enables step-by-step answer generation

### Example

Original Question:

```text
How does AI affect cybersecurity and software development?
```

Decomposed Queries:

```text
1. How is AI used in cybersecurity?
2. How is AI used in software engineering?
3. What are the risks of AI in cybersecurity?
4. What development tools use AI?
```

---

# Tech Stack

## Core Frameworks

* LangChain
* OpenAI API
* ChromaDB

## Python Libraries

* langchain
* langchain-openai
* langchain-community
* chromadb
* tiktoken
* bs4 (BeautifulSoup)

---

# Architecture

```text
User Query
     ↓
Generate Multiple Queries
     ↓
Retrieve Documents
     ↓
Combine + Rerank Results (RRF)
     ↓
Build Context
     ↓
LLM Generates Final Response
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

## 2. Install Dependencies

```bash
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
```

---

# Environment Variables

Create environment variables for OpenAI and LangSmith.

```python
import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'YOUR_LANGSMITH_KEY'

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
```

---

# Project Workflow

## Step 1 — Load Documents

The notebook uses web-based document loading with:

```python
WebBaseLoader
```

---

## Step 2 — Split Documents

Documents are chunked using:

```python
RecursiveCharacterTextSplitter
```

This improves embedding and retrieval efficiency.

---

## Step 3 — Create Embeddings

Embeddings are generated using OpenAI embedding models.

---

## Step 4 — Store in Vector Database

Document vectors are stored inside ChromaDB.

---

## Step 5 — Retrieve Relevant Documents

Retriever searches are performed using:

* Multi-query generation
* Query decomposition
* RAG Fusion

---

## Step 6 — Generate Final Answer

Retrieved context is passed into the LLM for grounded response generation.

---

# Key Concepts Explained

## What is RAG?

Retrieval-Augmented Generation combines:

1. Retrieval Systems → Search relevant documents
2. Large Language Models → Generate responses

This helps LLMs answer questions using external knowledge instead of relying only on training data.

---

## What is Reciprocal Rank Fusion (RRF)?

RRF is a reranking algorithm used to combine ranked retrieval results from multiple queries.

### Formula

```text
Score = Σ 1 / (rank + k)
```

Where:

* rank = position of document in search results
* k = smoothing constant

Documents appearing frequently across searches receive higher scores.

---
