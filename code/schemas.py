from pydantic import BaseModel  # Parent class for pydantic models (classes must inherit)
from pydantic import Field  # Add constraints to individual fields

##################
# REQUEST MODELS
##################
class QueryRequest(BaseModel):
    """Body of POST /query - Question sent to the RAG agent"""
    question: str = Field(..., min_length=1, max_length=1000)  # "..." makes <question> mandatory input from client
    num_results: int = Field(default=3, ge=1, le=10)

class IngestRequest(BaseModel):
    """Body of POST /ingest — tells the server where to find documents."""
    documents_path: str = Field(default="./documents")

##################
# RESPONSE MODELS
##################
class SourceDocument(BaseModel):
    """
    One retrieved chunk returned alongside the answer. This lets the client show users
    WHERE the answer came from, which builds trust and lets users verify the information.
    """
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    """Full response from POST /query"""
    question: str
    answer: str
    sources: list[SourceDocument]

class IngestResponse(BaseModel):
    """Stats returned after indexing documents."""
    documents_loaded: int
    chunks_created: int
    status: str  # e.g. success

class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str
    documents_loaded: bool




