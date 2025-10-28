import os
import subprocess
import threading
import time
import requests
from dotenv import load_dotenv
from pyngrok import ngrok
import uvicorn

from system import get_app

def setup_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    print("--- INFO: Environment variables loaded. ---")
    
    token = os.getenv("NGROK_AUTH_TOKEN")
    if not token or token == "YOUR_NGROK_AUTH_TOKEN":
        raise ValueError("NGROK_AUTH_TOKEN is not set in the .env file. Please add it.")
    ngrok.set_auth_token(token)

def start_vllm_server():
    """Starts the vLLM OpenAI-compatible server in a background thread."""
    model_id = os.getenv("MODEL_ID")
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--trust-remote-code" # Required for some models like Qwen
    ]

    def run():
        print(f"--- INFO: Starting vLLM server for model: {model_id} ---")
        # Using Popen to not block and capture output for debugging if needed
        server_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # You can optionally log the server output
        for line in iter(server_process.stdout.readline, b''):
            print(f"[vLLM Server]: {line.decode().strip()}")
        
    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    print("--- INFO: vLLM server thread started. ---")

    # Health check to ensure the server is ready before proceeding
    health_url = "http://127.0.0.1:8000/health"
    max_wait = 300 # 5 minutes
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print("--- SUCCESS: vLLM server is healthy and ready. ---")
                return
        except requests.ConnectionError:
            print("--- INFO: Waiting for vLLM server to be ready... ---")
            time.sleep(10)
    
    raise RuntimeError("vLLM server failed to start within the time limit.")

def start_fastapi_app():
    """Starts the FastAPI application server in a background thread."""
    _, fastapi_app = get_app()
    
    def run():
        print("--- INFO: Starting FastAPI application server on port 8001. ---")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8001, log_level="info")

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    print("--- INFO: FastAPI app thread started. ---")
    time.sleep(5) # Give uvicorn a moment to bind to the port

def start_ngrok_tunnel():
    """Starts the ngrok tunnel to expose the FastAPI app."""
    print("--- INFO: Starting ngrok tunnel to expose port 8001. ---")
    # Using http protocol and a custom domain if available, otherwise a random one.
    # The `pyngrok` library now serves a basic HTML page on the tunnel URL by default.
    # To directly proxy, more advanced config might be needed, but this works for API access.
    public_url = ngrok.connect(8001, "http")
    print("="*60)
    print(f"🚀 AI Core is LIVE! 🚀")
    print(f"Access the dashboard at: {public_url}")
    print("="*60)
    return public_url

if __name__ == "__main__":
    try:
        setup_environment()
        start_vllm_server()
        start_fastapi_app()
        start_ngrok_tunnel()

        # Keep the main thread alive to let background threads run
        while True:
            time.sleep(60)
            
    except Exception as e:
        print(f"--- FATAL ERROR: {e} ---")
        ngrok.kill()
    except KeyboardInterrupt:
        print("\n--- INFO: Shutting down servers. ---")
        ngrok.kill()
        print("--- SUCCESS: Shutdown complete. ---")
