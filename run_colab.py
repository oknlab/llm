"""
🚀 ENTERPRISE MULTI-AGENT ORCHESTRATION SYSTEM - ENTRY POINT
Elite architecture: Modular, scalable, production-ready
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
import json

# === CONFIGURATION ===
load_dotenv()
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# === GLOBAL STATE ===
class SystemState:
    """Thread-safe global state management"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.ngrok_url = None
        self.system_core = None
        self.active_connections = []
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "agents_executed": {},
            "startup_time": datetime.utcnow().isoformat()
        }

state = SystemState()

# === MODEL INITIALIZATION ===
def initialize_model():
    """Load Qwen3-1.7B model with optimization"""
    logger.info("🔄 Initializing Qwen3-1.7B model...")
    
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    
    try:
        # Load tokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with optimization
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        state.pipeline = pipeline(
            "text-generation",
            model=state.model,
            tokenizer=state.tokenizer,
            max_new_tokens=int(os.getenv("AGENT_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),
            device=device
        )
        
        logger.info(f"✅ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        return False

# === FASTAPI APPLICATION ===
app = FastAPI(
    title="Elite Multi-Agent Orchestration System",
    version="1.0.0",
    description="Enterprise-grade AI agent framework with RAG & Airflow"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENABLE_CORS") == "true" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard"""
    try:
        with open("dashbord.html", "r") as f:
            html_content = f.read()
        # Inject NGROK URL
        html_content = html_content.replace(
            "{{API_URL}}", 
            state.ngrok_url or f"http://localhost:{os.getenv('API_PORT', '8000')}"
        )
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard not found. Ensure dashbord.html exists.</h1>", status_code=404)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": state.pipeline is not None,
        "system_active": state.system_core is not None,
        "ngrok_url": state.ngrok_url,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    return JSONResponse(content=state.metrics)

@app.post("/agent/execute")
async def execute_agent(request: dict, background_tasks: BackgroundTasks):
    """Execute specific agent with task"""
    state.metrics["requests_total"] += 1
    
    try:
        if not state.system_core:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        agent_name = request.get("agent")
        task = request.get("task")
        
        if not agent_name or not task:
            raise HTTPException(status_code=400, detail="Missing agent or task")
        
        # Execute agent
        result = await state.system_core.execute_agent(agent_name, task)
        
        # Update metrics
        state.metrics["requests_success"] += 1
        state.metrics["agents_executed"][agent_name] = \
            state.metrics["agents_executed"].get(agent_name, 0) + 1
        
        return {
            "status": "success",
            "agent": agent_name,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        state.metrics["requests_failed"] += 1
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    state.active_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_json({
                "type": "metrics_update",
                "data": state.metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(2)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.active_connections.remove(websocket)

@app.post("/rag/query")
async def rag_query(request: dict):
    """RAG query endpoint"""
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Missing query")
        
        result = await state.system_core.rag_pipeline.query(query)
        
        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === NGROK SETUP ===
def setup_ngrok():
    """Initialize NGROK tunnel"""
    token = os.getenv("NGROK_AUTH_TOKEN")
    if not token or token == "your_ngrok_token_here":
        logger.warning("⚠️  NGROK token not configured. Running in local mode only.")
        return None
    
    try:
        ngrok.set_auth_token(token)
        port = int(os.getenv("API_PORT", "8000"))
        tunnel = ngrok.connect(port, bind_tls=True)
        url = tunnel.public_url
        logger.info(f"🌐 NGROK tunnel established: {url}")
        return url
    except Exception as e:
        logger.error(f"❌ NGROK setup failed: {e}")
        return None

# === MAIN EXECUTION ===
async def startup_event():
    """Application startup sequence"""
    logger.info("🚀 Starting Elite Multi-Agent System...")
    
    # Step 1: Initialize model
    if not initialize_model():
        logger.error("Model initialization failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Initialize system core
    from system import SystemCore
    state.system_core = SystemCore(state.pipeline, state.tokenizer)
    await state.system_core.initialize()
    
    # Step 3: Setup NGROK
    state.ngrok_url = setup_ngrok()
    
    logger.info("✅ System fully operational")
    logger.info(f"📊 Dashboard: {state.ngrok_url or 'http://localhost:8000'}")

@app.on_event("startup")
async def on_startup():
    await startup_event()

@app.on_event("shutdown")
async def on_shutdown():
    """Graceful shutdown"""
    logger.info("🛑 Shutting down system...")
    if state.system_core:
        await state.system_core.shutdown()
    ngrok.disconnect(state.ngrok_url)

def main():
    """Main entry point"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
