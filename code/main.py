# http://localhost:8000/docs <- Interactive Swagger UI


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
# CORSMiddleware lets the browser-based React front-end (served from a different
# origin, e.g. file:// or another port) call this API without being blocked by
# the browser's same-origin policy.
from fastapi.middleware.cors import CORSMiddleware
from schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, HealthResponse

from rag_engine import RAGEngine

# This runs ONCE when the module is imported. The same instance is reused for every request
rag_engine = RAGEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code that runs ONCE when the server starts and stops.

    How it works:
      1. Code BEFORE yield runs at server startup
      2. yield pauses — the server handles requests during this time
      3. Code AFTER yield runs at server shutdown

    We use it to load any previously indexed documents from disk,
    so the server can answer questions immediately without needing
    a fresh POST /ingest on every restart.
    """
    if rag_engine.load_existing_store():
        print("Loaded existing ChromaDB vector store from disk.")
    else:
        print("No existing vector store found.")
        print("POST to /ingest to index your documents first.")

    # yield = "server is now running and accepting requests"
    yield
    print("Server shutting down.")


app = FastAPI(
    title="RAG Agent API",
    description=("""A simple AI agent that answers questions using Retrieval-Augmented Generation (RAG). Built with 
        LangChain, Google Gemini, ChromaDB, and FastAPI."""),
    lifespan=lifespan,
)

# Allow the local React front-end to call the API. For this simple local demo,
# allow all origins; tighten allow_origins to your front-end URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the server is alive and documents are loaded.
    HTTP: GET /health
    Example response: {"status": "healthy", "documents_loaded": true}
    """
    return HealthResponse(status="healthy", documents_loaded=rag_engine.vector_store is not None)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Index .txt documents from a folder into ChromaDB.
    HTTP: POST /ingest
    Body: {"documents_path": "./documents"}

    Example response: {"documents_loaded": 1, "chunks_created": 8, "status": "success"}

    Call this once after starting the server, or again whenever your source documents change.
    You access fields as attributes: request.documents_path
    """
    try:
        result = rag_engine.ingest_documents(request.documents_path)
        return IngestResponse(**result)
    except ValueError as e:  # Client Error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # Server Error
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question and get a RAG-powered answer.

    HTTP: POST /query
    Body: {"question": "What is FastAPI?", "num_results": 3}

    Example response:
    {"question": "What is FastAPI?",
    "answer": "FastAPI is a modern Python web framework...",
    "sources": [{"content": "FastAPI is a modern, fast...", "metadata": {"source": "documents/sample.txt"}}]}
    """
    try:
        result = rag_engine.query(
            question=request.question,
            k=request.num_results,
        )
        return QueryResponse(**result)

    except ValueError as e:  # Client Error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # Server Error
        raise HTTPException(status_code=500, detail=f"Query Failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
