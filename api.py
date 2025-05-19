import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import traceback
from typing import Optional, Dict, List, Any

# Import your RAG class
from rag.core import EquipmentInspectionRAG

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
api_logger = logging.getLogger("FastAPIApi")

# --- RAG Initialization ---
EXCEL_DATA_PATH = "data/equipment_inspection_data.xlsx"
rag_system: Optional[EquipmentInspectionRAG] = None

try:
    api_logger.info(f"Initializing RAG system from {EXCEL_DATA_PATH}...")
    rag_system = EquipmentInspectionRAG(EXCEL_DATA_PATH)
    api_logger.info("RAG system setup complete. Creating graph...")
    rag_system.create_graph()
    if rag_system and rag_system.graph:
        api_logger.info("RAG graph created successfully.")
    else:
        api_logger.error("Failed to create RAG graph.")
except Exception as e:
    api_logger.error(f"Failed to initialize RAG system: {e}")
    traceback.print_exc()

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Request Models ---
class QueryRequest(BaseModel):
    query: str

class ResolutionRequest(BaseModel):
    punch_id: str
    new_status: str = "1"
    resolution_text: str = ""
    revision_timestamp: str

# --- Routes ---

@app.get("/")
async def read_root():
    """Serve the main index.html file from static/ directory."""
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        api_logger.error(f"index.html not found at {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    try:
        with open(index_path) as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        api_logger.error(f"Error serving index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal server error serving index.html")

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Handle a user question via RAG pipeline."""
    api_logger.info(f"Received query request: {request.query}")
    if not rag_system or not rag_system.graph:
        detail = "RAG system not initialized or graph unavailable."
        if rag_system:
            if rag_system.df is None:
                detail = "RAG system initialized but data loading failed."
            elif rag_system.llm is None:
                detail = "RAG system initialized but LLM connection failed."
            elif rag_system.retriever is None:
                detail = "RAG system initialized but vectorstore/retriever failed."
            elif rag_system.graph is None:
                detail = "RAG system initialized but graph creation failed."
        raise HTTPException(status_code=503, detail=detail)

    try:
        answer = rag_system.query(request.query)
        api_logger.info("Query processed successfully.")
        return {"answer": answer}
    except Exception as e:
        api_logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")

@app.get("/punches")
async def get_punches(disc: str = "", item_type: str = "", punch_status: str = ""):
    """Fetch punches filtered by discipline, item type, and punch status."""
    api_logger.info(f"Fetching punches: disc={disc}, item_type={item_type}, punch_status={punch_status}")
    if not rag_system or rag_system.df is None:
        detail = "RAG system not initialized or data unavailable."
        raise HTTPException(status_code=503, detail=detail)

    try:
        punches = rag_system.filter_punches(disc=disc, item_type=item_type, punch_status=punch_status)
        api_logger.info(f"Returning {len(punches)} punches.")
        return punches
    except Exception as e:
        api_logger.error(f"Error filtering punches: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while fetching punches.")

@app.post("/punches/resolve")
async def resolve_punch(request: ResolutionRequest):
    """Mark a punch as resolved and add revision data."""
    api_logger.info(f"Resolving punch: {request.punch_id}")
    if not rag_system or rag_system.df is None:
        detail = "RAG system not initialized or data unavailable."
        raise HTTPException(status_code=503, detail=detail)

    try:
        rag_system.add_punch_resolution(
            punch_id=request.punch_id,
            new_status=request.new_status,
            resolution_text=request.resolution_text,
            revision_timestamp=request.revision_timestamp  # <-- FIXED ARG NAME
        )
        return {"status": "success"}
    except ValueError as e:
        api_logger.warning(f"Resolution validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        api_logger.error(f"Error adding punch resolution: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while submitting resolution.")

@app.get("/health")
async def health_check():
    """Simple health check for backend + RAG status."""
    if rag_system and rag_system.initialized_properly:
        return {"status": "ok"}
    return {"status": "degraded", "detail": "RAG system not fully initialized."}

@app.post("/reload")
async def reload_rag_system():
    """Reload the RAG system from disk."""
    global rag_system
    try:
        rag_system = EquipmentInspectionRAG(EXCEL_DATA_PATH)
        rag_system.create_graph()
        return {"status": "reloaded"}
    except Exception as e:
        api_logger.error(f"Failed to reload RAG system: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
