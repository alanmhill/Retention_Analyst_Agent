# 🧠 Agentic HR Retention Analytics Platform

An end-to-end agentic analytics system that combines:

-   Local LLM orchestration (DeepSeek-R1 via Ollama)
-   LangChain-based model abstraction
-   Vector retrieval (Qdrant)
-   Deterministic analytics tooling (Pandas)
-   FastAPI backend
-   Streamlit interactive UI
-   Fully Dockerized infrastructure

------------------------------------------------------------------------

## 🎯 Problem Statement

Enterprise HR teams often rely on static dashboards or manual analysis
to understand attrition drivers. These approaches lack:

-   Natural language querying
-   Context-aware insights
-   Automated executive summaries
-   Real-time analytical reasoning

This system enables users to:

1.  Upload HR datasets (CSV/XLSX)
2.  Ask natural language questions
3.  Automatically compute retention analytics
4.  Generate structured results
5.  Produce executive-level summaries via a local LLM

------------------------------------------------------------------------

## 🏗 Architecture

    Streamlit UI
            ↓
    FastAPI Backend
            ↓
    LangChain (ChatOllama)
            ↓
    DeepSeek-R1 (Local LLM via Ollama)
            ↓
    Retention Analytics Tool (Pandas)
            ↓
    Qdrant (Vector Database for RAG)

------------------------------------------------------------------------

## ⚙️ How It Works

### 1️⃣ Dataset Upload

-   Dataset is uploaded via Streamlit
-   Stored on disk
-   Column profiles embedded via `nomic-embed-text`
-   Stored in Qdrant for retrieval

### 2️⃣ Question Handling

When a user asks a question:

1.  The question is embedded
2.  Relevant dataset metadata is retrieved from Qdrant
3.  Deterministic analytics engine computes:
    -   Overall attrition rate
    -   Segment attrition (Department, JobRole, OverTime, etc.)
    -   Numeric deltas (Engagement, Income, Tenure)
    -   Tenure band analysis
4.  Results are passed to DeepSeek-R1 via LangChain
5.  LLM generates executive-level narrative summary

------------------------------------------------------------------------

## 📊 Example Output

The system returns:

-   KPI cards (overall attrition, highest-risk department)
-   Department attrition bar chart
-   Executive summary with quantified metrics
-   Structured JSON analysis
-   RAG context used for reasoning

Example summary excerpt:

> Overall attrition rate stands at 16.0%. Operations exhibits the
> highest department-level attrition at 21.2% (n=52). Employees working
> overtime demonstrate a 2.7x higher attrition rate compared to
> non-overtime staff. Recommended actions include workload balancing
> pilots and engagement pulse checks.

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

-   Docker Desktop
-   NVIDIA drivers (optional for GPU acceleration)

### Run the Full Stack

``` bash
docker compose up -d --build
```

This will: - Start Qdrant - Start Ollama - Automatically pull required
models - Start FastAPI backend - Start Streamlit UI

### Access

-   API: http://localhost:8000
-   Streamlit UI: http://localhost:8501

------------------------------------------------------------------------

## 🧠 Why This Is Agentic

This system demonstrates:

-   Tool-based reasoning (deterministic analytics functions)
-   Retrieval Augmented Generation (RAG)
-   LLM orchestration via LangChain
-   Separation of computation vs narrative synthesis
-   Model abstraction (swappable LLM backend)
-   Infrastructure reproducibility via Docker

The LLM interprets structured tool outputs rather than computing
analytics directly, ensuring deterministic correctness while preserving
flexible language generation.

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python 3.11
-   FastAPI
-   Streamlit
-   LangChain
-   Ollama
-   DeepSeek-R1
-   Qdrant
-   Pandas
-   Docker Compose

------------------------------------------------------------------------

## 👨‍💻 Author Contribution

Designed and implemented:

-   Full backend architecture
-   Vector embedding and retrieval logic
-   Retention analytics computation engine
-   LangChain LLM orchestration
-   Dockerized infrastructure
-   Interactive Streamlit UI
-   Model auto-initialization via Docker
-   Error handling and serialization safeguards

------------------------------------------------------------------------

## 📬 Contact

Available for walkthrough or technical discussion upon request.
