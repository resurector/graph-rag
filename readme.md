# GraphRAG: Neo4j + LangChain Retrieval-Augmented Generation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](#prerequisites)
[![Neo4j 4.x|5.x](https://img.shields.io/badge/Neo4j-4.x%20%7C%205.x-008CC1.svg)](https://neo4j.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API%20Key-lightgrey)](https://platform.openai.com/)

**GraphRAG** is a Python application that:
- Ingests documents (PDF, DOCX, or TXT),
- Splits them into chunks,
- Embeds chunks with OpenAI (or Azure) embeddings,
- Stores results in a Neo4j database,
- Performs retrieval-augmented generation (RAG) via an LLM to answer questions from the ingested documents.

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repo](#1-clone-the-repo)
  - [Create a Virtual Environment](#2-create-a-virtual-environment-optional)
  - [Install Dependencies](#3-install-dependencies)
  - [Set Environment Variables](#4-set-environment-variables)
- [Usage](#usage)
  - [Start Neo4j](#start-neo4j)
  - [Run the App](#run-the-app)
  - [Process Documents](#process-documents)
  - [Ask Questions](#ask-questions)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author--Contact](#author--contact)
- [Further-Reading](#further-reading)

---

## Features

1. **Document Upload**: Upload a PDF, DOCX, or TXT file.  
2. **Chunking & Embeddings**: Splits text into segments and embeds them with OpenAIEmbeddings.  
3. **Neo4j Storage**: Each chunk is stored in a Neo4j graph with a unique identifier.  
4. **RAG Chat Interface**: Ask a question; the app retrieves relevant chunks from the database, then uses a GPT-like model to generate an answer.  
5. **Gradio UI**: Simple web-based interface to handle both document processing and Q&A.

---

## Prerequisites

- **Python 3.8+**
- **Neo4j 4.x or 5.x** (with Bolt protocol enabled)  
  Make sure you have a running Neo4j instance at `bolt://localhost:7687` or set `NEO4J_URI` to your custom URL.
- **OpenAI API Key** (if using OpenAI embeddings and ChatOpenAI)  
  Sign up at [OpenAI](https://platform.openai.com/).

---
## Installation

### 1. Clone the Repo
```bash
git clone https://github.com/resurector/graph-rag.git
cd graph-rag


---


### 2. Create a Virtual Environment (Optional)
```bash
python -m venv venv
# For Linux/Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate


---
### 3. Install Dependencies
```bash
pip install -r requirements.txt


---
**Repository Structure**
```bash
graph-rag/
├── graphrag.py         # Main script with GraphRAGProcessor & Gradio UI
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables (optional)
└── README.md            # This file






