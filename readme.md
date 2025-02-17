# GraphRAG: Neo4j + LangChain Retrieval-Augmented Generation



[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](#prerequisites)
[![Neo4j 4.x|5.x](https://img.shields.io/badge/Neo4j-4.x%20%7C%205.x-008CC1.svg)](https://neo4j.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API%20Key-lightgrey)](https://platform.openai.com/)

## 📖 Learn More

For a detailed walkthrough on building a **GraphRAG application** with **Neo4j, OpenAI, and Gradio**, check out my Medium article:

👉 **[Building a GraphRAG Application with Neo4j, OpenAI, and Gradio](https://medium.com/@rhameed79/building-a-graphrag-application-with-neo4j-openai-and-gradio-d57f6246f9fe)**

This article covers the step-by-step process of implementing a **retrieval-augmented generation (RAG) system** using a **knowledge graph**, enhancing LLM responses with structured data.

Happy building! 🚀


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
```

---


### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
# For Linux/Mac:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```

---
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a .env file (or export them manually) with the following:
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

OPENAI_API_KEY=your_openai_api_key
PROVIDER_TYPE=openai
```

Adjust values to match your Neo4j credentials.

---

## Usage

### Start Neo4j
Make sure your Neo4j instance is running locally or is accessible via the URL you specified in your `.env`.

### Run the App

```bash
python Neo4j_app.py
```

By default, it will launch a Gradio interface at http://127.0.0.1:7860.

If share=True is set in the code, you’ll also get a public shareable link.


## Process Documents
1. Go to the Process Document tab in the web interface.
2. Upload a PDF, DOCX, or TXT file.
3. Adjust chunk size/overlap if desired.
4. Click Process File. This ingests the document into Neo4j.


## Ask Questions
1. Switch to the Chat tab.
2. Enter a question. The system will retrieve relevant chunks from Neo4j, then pass them to a GPT-like model (OpenAI) for an answer.
3. The result, plus chunk matches, are displayed in the interface.

###
**Repository Structure**
---
```bash
graph-rag/
├── Neo4j_app.py         # Main script with GraphRAGProcessor & Gradio UI
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables (optional)
└── README.md            # This file
```

---

## Troubleshooting

Neo4j Connection: If you get “Could not connect to Neo4j,” ensure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are correct and that Neo4j is running.
OpenAI Rate Limits: If you run many queries, you may hit OpenAI’s rate limit. Use an appropriate plan or throttle your requests.
Node Already Exists Error: If you see a Node already exists error, confirm your code sets a unique id for each document upload, or remove the uniqueness constraint if desired.

## Contributing
Contributions and pull requests are welcome! Feel free to open an issue for any bug reports or feature requests. To contribute:

## Fork the repository
Create a new branch (git checkout -b feature/some-feature)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/some-feature)
Open a Pull Request
---

#### License
This project is licensed under the MIT License. Feel free to reuse and adapt it for your own use cases.

## Author / Contact
Created by resurector.

---
## Further Reading

- **LangChain**: [LangChain Docs](https://langchain.readthedocs.io/)
- **Neo4j**: [Neo4j Official Site](https://neo4j.com/)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/)



