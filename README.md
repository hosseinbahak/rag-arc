# Equipment Inspection RAG System

This system manages equipment inspection checklists and punch lists using RAG (Retrieval-Augmented Generation) powered by LangGraph and LangChain.

## 🔧 Features

- ✅ Natural language query (English + فارسی)
- ✅ Punch record tracking with revision history
- ✅ Checklist generation by item type
- ✅ Vector-based semantic search
- ✅ FastAPI backend & web UI

---

## 📂 Directory Overview

- `main.py`: CLI test script for RAG queries.
- `api.py`: FastAPI RESTful backend for integration.
- `rag/core.py`: Core logic of the RAG system.
- `rag/schema.py`: Pydantic schemas.
- `data/`: Input Excel data.
- `static/index.html`: User interface.

---

## ⚙️ Setup

### 1. Clone the Repo

```bash
git clone https://your-git-repo-url.git
cd equipment_inspection_rag



## Run project
```bash
export OPENAI_API_KEY="your-api-key"
python3 main.py