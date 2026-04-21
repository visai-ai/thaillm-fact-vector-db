"""
Ingest in-memory vectors and start server
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from med_fact_retriever.data import MedFactDatabase
from med_fact_retriever.indexer import MedFactVectorIndexer
from med_fact_retriever.retriever import MedFactRetriever
from med_fact_retriever.data_model import RetrievedFacts, InputQuery
from med_fact_retriever.settings import AppSettings

if not load_dotenv():
    print(".env is not loaded. This is okay if you know what you're doing.")


logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("medfact.api")


API_KEY = os.getenv("API_KEY")
_auth_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(
    cred: Optional[HTTPAuthorizationCredentials] = Security(_auth_scheme),
    key: Optional[str] = Security(_api_key_header),
):
    # If no API key configured, auth is disabled.
    if not API_KEY:
        return
    candidate = None
    if cred and cred.scheme.lower() == "bearer":
        candidate = cred.credentials
    if not candidate and key:
        candidate = key
    if candidate != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AppSettings.from_env()
    logger.info(f"Loaded settings: {cfg}")

    # Ensure LanceDB base folder exists; the indexer manages persist_dir itself
    os.makedirs(os.path.join(cfg.cache_dir, MedFactVectorIndexer.DEFAULT_DATA_DIR), exist_ok=True)

    # Load dataset
    db = MedFactDatabase(cache_dir=cfg.cache_dir, include_golden_fact=cfg.include_golden_fact)
    nodes = db.export_text_node()
    logger.info("Loaded %d facts from dataset", len(db))

    # Build or load index
    indexer = MedFactVectorIndexer(
        cache_dir=cfg.cache_dir,
        nprobes=cfg.nprobes,
    )

    if not indexer.is_indexed():
        logger.info("No valid index found. Ingesting %d nodes…", len(nodes))
        indexer.ingest(nodes, show_progress=True)
        logger.info("Index persisted to %s", indexer.persist_dir)
    else:
        logger.info("Loaded persisted index from %s", indexer.persist_dir)

    # Prepare retriever (+ reranker)
    base_retriever = indexer.get_retriever(top_k=max(30, cfg.rerank_top_n))
    retriever = MedFactRetriever(
        retriever=base_retriever,
        use_reranker=cfg.use_reranker,
        top_n=cfg.rerank_top_n,
    )

    # share state
    app.state.db = db
    app.state.indexer = indexer
    app.state.retriever = retriever
    logger.info("Startup complete. Device: %s", "cuda" if indexer.is_cuda_available() else "cpu")

    yield  # ---- application runs here ----

    # ---- shutdown ----
    logger.info("Shutting down MedFact API")
    app.state.retriever = None
    app.state.indexer = None
    app.state.db = None
    logger.info("Shutdown complete.")

app = FastAPI(
    title="MedFact Retriever API",
    version="1.0.0",
    description="Search medical facts and return the best-matching fact.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=RetrievedFacts, dependencies=[Depends(require_api_key)])
def query(query: InputQuery) -> RetrievedFacts:
    if app.state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")

    results = app.state.retriever(**query.model_dump())
    if not results:
        raise HTTPException(status_code=404, detail="No facts found")

    return results
