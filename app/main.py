import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
import math
import json
import asyncio
from typing import Union
import httpx
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# -----------------------------
# Config
# -----------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-r1:8b")

llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

exec_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior People Analytics consultant. Use only the provided metrics. Do not invent data."),
    ("human",
     """User question:
{question}

Computed metrics (JSON):
{analysis_json}

Return:
- 6 to 10 bullet points total
- Start with overall attrition rate
- Then strongest segmentation insights (Department/JobRole/OverTime/etc.) including counts (n)
- Then 1-2 numeric deltas if meaningful
- Then recommended actions to test + KPIs to monitor weekly
- Skip null/empty metrics
""")
])

# Embedding model served by Ollama
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Persist uploaded datasets here (mount a docker volume to this path)
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use a collection name tied to the embedding model so you don't collide later
COLLECTION = f"datasets__{EMBED_MODEL.replace('-', '_')}"

app = FastAPI(title="Agentic Analytics MVP")

qdrant = QdrantClient(url=QDRANT_URL)


# -----------------------------
# Helpers
# -----------------------------
def dataset_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.csv"


def df_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [],
    }

    for col in df.columns:
        s = df[col]
        col_info = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100.0),
            "nunique": int(s.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            col_info.update(
                {
                    "min": float(desc.get("min", float("nan"))),
                    "max": float(desc.get("max", float("nan"))),
                    "mean": float(desc.get("mean", float("nan"))),
                    "std": float(desc.get("std", float("nan"))),
                }
            )

        profile["columns"].append(col_info)

    return profile


def to_chunks(dataset_id: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    chunks.append(
        {
            "dataset_id": dataset_id,
            "type": "dataset_summary",
            "text": f"Dataset {dataset_id}: {profile['rows']} rows, {profile['cols']} columns.",
            "meta": {"rows": profile["rows"], "cols": profile["cols"]},
        }
    )

    for c in profile["columns"]:
        text = (
            f"Column {c['name']} dtype={c['dtype']} missing={c['missing']} "
            f"({c['missing_pct']:.2f}%) nunique={c['nunique']}."
        )
        if "mean" in c:
            text += (
                f" Numeric stats: min={c['min']}, max={c['max']}, "
                f"mean={c['mean']}, std={c['std']}."
            )

        chunks.append(
            {
                "dataset_id": dataset_id,
                "type": "column_profile",
                "text": text,
                "meta": c,
            }
        )

    return chunks


async def ollama_embed(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        # If this fails, you'll get a useful error from Ollama
        r.raise_for_status()
        data = r.json()
        return data["embedding"]


async def ensure_collection(vector_size: int) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    # If it exists, verify dimension matches (prevents hard-to-debug 400s)
    info = qdrant.get_collection(collection_name=COLLECTION)
    # Qdrant can store vector config in different forms; this is the common path.
    current_size = info.config.params.vectors.size  # type: ignore[attr-defined]
    if current_size != vector_size:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Qdrant collection '{COLLECTION}' has vector size {current_size}, "
                f"but embedding model '{EMBED_MODEL}' returned size {vector_size}. "
                f"Fix by deleting the Qdrant volume or using a new COLLECTION name."
            ),
        )


def retention_driver_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    if "Attrition" not in df.columns:
        raise ValueError("No 'Attrition' column found in dataset.")

    # Normalize Attrition to yes/no
    attr = df["Attrition"].astype(str).str.strip().str.lower()
    work = df.copy()
    work["_attr_yes"] = (attr == "yes").astype(int)

    overall_rate = float(work["_attr_yes"].mean())
    results: Dict[str, Any] = {"overall_attrition_rate": overall_rate}

    # Segment attrition rates
    segment_cols = ["Department", "JobRole", "OverTime", "PerformanceScore", "EducationLevel"]
    seg: Dict[str, Any] = {}
    for col in segment_cols:
        if col in work.columns:
            by = (
                work.groupby(col)["_attr_yes"]
                .agg(["mean", "count"])
                .sort_values("mean", ascending=False)
            )
            seg[col] = [
                {"group": str(idx), "attrition_rate": float(row["mean"]), "n": int(row["count"])}
                for idx, row in by.head(8).iterrows()
            ]
    results["segment_attrition"] = seg

    # Numeric deltas: attriters vs stayers
    numeric_cols = ["YearsAtCompany", "MonthlyIncome", "EngagementScore", "RemoteWorkPct", "PTOUsed"]
    deltas: Dict[str, Any] = {}
    for col in numeric_cols:
        if col in work.columns and pd.api.types.is_numeric_dtype(work[col]):
            a = work.loc[work["_attr_yes"] == 1, col].dropna()
            b = work.loc[work["_attr_yes"] == 0, col].dropna()
            if len(a) > 5 and len(b) > 5:
                deltas[col] = {
                    "mean_attriters": float(a.mean()),
                    "mean_stayers": float(b.mean()),
                    "difference": float(a.mean() - b.mean()),
                }
    results["numeric_deltas"] = deltas

    # Tenure bands
    if "YearsAtCompany" in work.columns and pd.api.types.is_numeric_dtype(work["YearsAtCompany"]):
        bins = [0, 1, 2, 5, 10, 100]
        labels = ["<1y", "1-2y", "2-5y", "5-10y", "10y+"]
        work["_tenure_band"] = pd.cut(work["YearsAtCompany"], bins=bins, labels=labels, right=False)

        by_band = work.groupby("_tenure_band")["_attr_yes"].agg(["mean", "count"]).sort_index()
        results["tenure_bands"] = [
            {"band": str(idx), "attrition_rate": float(row["mean"]), "n": int(row["count"])}
            for idx, row in by_band.iterrows()
        ]

    return results



JsonLike = Union[dict, list, str, int, float, bool, None]

def sanitize_for_json(obj: JsonLike) -> JsonLike:
    """
    Recursively convert NaN/Inf floats into None so JSON serialization won't fail.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def format_top3(analysis: Dict[str, Any]) -> Dict[str, Any]:
    seg = analysis.get("segment_attrition", {})
    dept = seg.get("Department", [])[:3]
    role = seg.get("JobRole", [])[:3]
    return {
        "top_departments": dept,
        "top_job_roles": role,
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
        "data_dir": str(DATA_DIR),
        "embed_model": EMBED_MODEL,
    }


@app.get("/health")
async def health():
    results: Dict[str, Any] = {}

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            results["ollama"] = {"ok": r.status_code == 200, "status_code": r.status_code}
    except Exception as e:
        results["ollama"] = {"ok": False, "error": str(e)}

    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{QDRANT_URL}/collections")
            results["qdrant"] = {"ok": r.status_code == 200, "status_code": r.status_code}
    except Exception as e:
        results["qdrant"] = {"ok": False, "error": str(e)}

    results["ok"] = bool(results.get("ollama", {}).get("ok")) and bool(results.get("qdrant", {}).get("ok"))
    return results


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Please upload a .csv or .xlsx file.")

    raw = await file.read()

    # Parse
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(raw))
        else:
            df = pd.read_excel(BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    dataset_id = str(uuid.uuid4())

    # Persist dataset (store as CSV regardless of original input)
    try:
        df.to_csv(dataset_path(dataset_id), index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist dataset: {e}")

    # Profile -> chunks
    profile = df_profile(df)
    chunks = to_chunks(dataset_id, profile)

    # Embed one chunk to infer embedding dimension & ensure Qdrant collection
    first_vec = await ollama_embed(chunks[0]["text"])
    await ensure_collection(vector_size=len(first_vec))

    # Upsert embedded chunks
    points: List[PointStruct] = []
    for ch in chunks:
        vec = await ollama_embed(ch["text"])
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "dataset_id": ch["dataset_id"],
                    "type": ch["type"],
                    "text": ch["text"],
                    "meta": ch["meta"],
                    "filename": file.filename,
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "saved_to": str(dataset_path(dataset_id)),
        "profile": {"rows": profile["rows"], "cols": profile["cols"], "n_chunks": len(chunks)},
    }


class AskRequest(BaseModel):
    dataset_id: str
    question: str
    top_k: int = 6


@app.post("/ask")
async def ask(req: AskRequest):
    # Embed question
    qvec = await ollama_embed(req.question)

    # Retrieve relevant context from Qdrant (filter by dataset_id)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=req.top_k,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="dataset_id",
                    match=MatchValue(value=req.dataset_id),
                )
            ]
        ),
    )

    contexts = [
        {
            "score": float(h.score),
            "text": h.payload.get("text"),
            "meta": h.payload.get("meta"),
            "type": h.payload.get("type"),
        }
        for h in hits
    ]

    # Load dataset from disk and compute retention analytics
    path = dataset_path(req.dataset_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found on server (DATA_DIR). Re-upload it.",
        )

    df = pd.read_csv(path)

    try:
        analysis = retention_driver_analysis(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {e}")

    # Your structured short answer (top 3)
    summary = format_top3(analysis)

    # Build prompt for executive summary
    # Use json.dumps so the LLM gets a clean, deterministic representation
    analysis_json = json.dumps(sanitize_for_json(analysis), indent=2)

    prompt_value = exec_prompt.format_prompt(
        question=req.question,
        analysis_json=analysis_json,
    )

    # LangChain's invoke() is sync; run it off the event loop
    def _run_llm():
        return llm.invoke(prompt_value.to_messages()).content

    try:
        executive_summary = await asyncio.to_thread(_run_llm)
    except Exception as e:
        # Don't fail the endpoint if LLM call fails; return analysis anyway
        executive_summary = f"(LLM summary unavailable: {e})"

    return sanitize_for_json({
        "dataset_id": req.dataset_id,
        "question": req.question,
        "answer": summary,
        "executive_summary": executive_summary,
        "analysis": analysis,
        "retrieved": contexts,
    })