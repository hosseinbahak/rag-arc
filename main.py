import webbrowser
import uvicorn
import threading
from rag.core import EquipmentInspectionRAG

# Preload RAG system (optional)
rag = EquipmentInspectionRAG("data/equipment_inspection_data.xlsx")
rag.create_graph()

# Function to launch FastAPI
def run_api():
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)

# Function to open browser
def open_browser():
    webbrowser.open_new("http://localhost:8081")

if __name__ == "__main__":
    # Start API in a thread
    threading.Thread(target=run_api, daemon=True).start()
    
    # Open web UI
    open_browser()
    
    print("RAG backend is running at http://localhost:8008")
    print("UI is opening at http://localhost:8081")
    
    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n🛑 Stopping server.")
