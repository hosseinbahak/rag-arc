# rag/schema.py
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

class PunchResolution(BaseModel):
    punch_id: str
    resolution_text: str
    revision: str
    new_status: Optional[str] = "1"
