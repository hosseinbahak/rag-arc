# api.py - Add some prints to see when endpoints are hit
# api.py
from fastapi import FastAPI, HTTPException
from rag.core import EquipmentInspectionRAG
from rag.schema import QueryRequest, QueryResponse, PunchResolution
from typing import List, Dict, Any

# Initialize RAG system globally in the API module
# This instance will be used by all incoming requests
try:
    print("API: Initializing EquipmentInspectionRAG...")
    # Make sure this path is correct relative to where api.py is run from (typically project root)
    rag = EquipmentInspectionRAG("data/equipment_inspection_data.xlsx")
    # Create graph here too, as main.py might not run first in some deployment scenarios
    rag.create_graph()
    print("API: RAG system initialized and graph created.")
except Exception as e:
    print(f"API: Failed to initialize RAG system: {e}")
    rag = None # Set rag to None if initialization fails

app = FastAPI()

@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    print(f"API: Received POST /query with query: {req.query}")
    if rag is None:
        raise HTTPException(status_code=500, detail="RAG system is not initialized.")
    try:
        answer = rag.query(req.query)
        print("API: Query processed, returning answer.")
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"API: Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during query processing: {e}")


@app.post("/punches/resolve")
def resolve_punch(data: PunchResolution):
    print(f"API: Received POST /punches/resolve for punch_id: {data.punch_id}, status: {data.new_status}, revision: {data.revision}")
    if rag is None:
         raise HTTPException(status_code=500, detail="RAG system is not initialized.")
    try:
        rag.add_punch_resolution(
            punch_id=data.punch_id,
            new_status=data.new_status,
            resolution_text=data.resolution_text,
            revision=data.revision
        )
        print("API: Punch resolution processed.")
        return {"status": "success"}
    except ValueError as e:
        print(f"API: Error adding resolution (ValueError): {e}")
        raise HTTPException(status_code=404, detail=str(e)) # 404 if punch not found
    except Exception as e:
        print(f"API: Error adding resolution: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during resolution: {e}")


@app.get("/punches", response_model=List[Dict[str, Any]]) # Add response model hint
def list_punches(discipline: str = "", item_type: str = "", punch_status: str = ""):
    print(f"API: Received GET /punches with filters - Disc:'{discipline}', ItemType:'{item_type}', Status:'{punch_status}'")
    if rag is None:
         raise HTTPException(status_code=500, detail="RAG system is not initialized.")
    try:
        df_filtered_records = rag.filter_punches(Disc=discipline, ItemType=item_type, Punch_Status=punch_status)
        print(f"API: Filtered punches, returning {len(df_filtered_records)} records.")
        return df_filtered_records
    except Exception as e:
        print(f"API: Error filtering punches: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during filtering: {e}")

# The main.py will import this `app` instance and potentially mount it.
# The previous main.py includes `app.include_router(api_app.router)`, which works.