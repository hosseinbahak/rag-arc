# main.py
import webbrowser
import uvicorn
import threading
import os
import time # Import time

# Function to launch FastAPI
def run_api():
    # Set environment variables if needed (e.g., for development)
    # os.environ['OPENAI_API_KEY'] = 'YOUR_ACTUAL_OPENAI_API_KEY' # Uncomment and set if not set globally

    # Use api:app to point to the FastAPI application defined in api.py
    # Add --log-level debug to see more server startup info
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False, log_level="info") # Changed log_level to info for more output

# Function to open browser
def open_browser():
    # Give the server a significant moment to start
    print("Waiting 10 seconds for server to initialize...")
    time.sleep(10) # <-- Increased delay
    print("Attempting to open browser...")
    try:
        webbrowser.open_new("http://localhost:8000") # Open the FastAPI server address
    except Exception as e:
        print(f"Failed to open browser automatically: {e}")
        print("Please open your web browser and go to http://localhost:8000 manually.")


if __name__ == "__main__":
    # Check for OPENAI_API_KEY before starting
    if not os.getenv("OPENAI_API_KEY"):
        print("🛑 Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script.")
        print("Example: export OPENAI_API_KEY='your-key' (Linux/macOS) or set OPENAI_API_KEY=your-key (Windows)")
    else:
        print("✅ OPENAI_API_KEY is set.")
        # Start API in a thread
        print("Starting FastAPI backend...")
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        print("FastAPI server is starting at http://localhost:8000")
        # Removed redundant print about UI opening here as open_browser has its own print

        # Open web UI (after delay)
        open_browser()

        print("\nPress Ctrl+C to stop the server.")

        # Keep the main thread alive
        try:
            # A simple way to keep main thread alive without consuming CPU
            # You could also join the api_thread if it weren't daemon, but daemon is better for exiting with Ctrl+C
            # Check if the thread is still alive periodically
            while True:
                 if not api_thread.is_alive():
                      print("🛑 FastAPI thread stopped unexpectedly.")
                      break # Exit the loop if thread dies
                 time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Stopping server.")