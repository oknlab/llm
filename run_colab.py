"""
╔═══════════════════════════════════════════════════════════════════════╗
║  COLAB LAUNCHER - FastAPI Server with NGROK Tunneling                 ║
║  Orchestrates: API Server + WebSocket + Dashboard + Monitoring        ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import nest_asyncio

# Allow nested event loops in Colab
nest_asyncio.apply()

# FastAPI & Uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
import uvicorn

# Monitoring
from prometheus_client import make_asgi_app, generate_latest
from starlette.responses import Response

# NGROK
from pyngrok import ngrok, conf

# Logging
from loguru import logger

# System imports
from system import (
    orchestrator,
    TaskRequest,
    TaskResponse,
    AgentType,
    system_health_check,
    settings,
    WorkflowDAG
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configure logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add("logs/system_{time}.log", rotation="500 MB", retention="10 days", compression="zip")

# Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """API key validation"""
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="🔷 Elite Multi-Agent System",
    description="Enterprise AI Orchestration with RAG, LangGraph & Multi-Agent Coordination",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

manager = ConnectionManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    dashboard_path = Path("dashboard.html")
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<h1>Dashboard not found. Ensure dashboard.html exists.</h1>", status_code=404)

@app.get("/api/health")
async def health():
    """System health check"""
    return await system_health_check()

@app.post("/api/task", response_model=TaskResponse)
async def execute_task(task: TaskRequest, api_key: str = Depends(verify_api_key)):
    """Execute agent task"""
    logger.info(f"Received task: {task.task_id}")
    
    # Broadcast task start
    await manager.broadcast({
        "event": "task_started",
        "task_id": task.task_id,
        "description": task.description
    })
    
    try:
        result = await orchestrator.execute_task(task)
        
        # Broadcast task completion
        await manager.broadcast({
            "event": "task_completed",
            "task_id": task.task_id,
            "result": result.dict()
        })
        
        return result
    
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        await manager.broadcast({
            "event": "task_failed",
            "task_id": task.task_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {
                "type": agent_type.value,
                "status": "active",
                "specialty": agent.specialty
            }
            for agent_type, agent in orchestrator.agents.items()
        ]
    }

@app.get("/api/dag")
async def get_dag():
    """Get workflow DAG definition"""
    return WorkflowDAG.create_agent_pipeline()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or process
            await websocket.send_json({"status": "received", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# NGROK TUNNEL
# ============================================================================

def setup_ngrok(port: int = 8000) -> str:
    """Setup NGROK tunnel"""
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if ngrok_token:
        conf.get_default().auth_token = ngrok_token
        logger.info("NGROK token configured")
    else:
        logger.warning("NGROK_AUTH_TOKEN not set. Using free tier (may be unstable).")
    
    # Terminate existing tunnels
    ngrok.kill()
    
    # Create tunnel
    public_url = ngrok.connect(port, bind_tls=True)
    logger.success(f"🌐 NGROK Tunnel: {public_url}")
    
    return public_url.public_url

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("🚀 Starting Elite Multi-Agent System...")
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True)
    
    # Warm up models
    logger.info("Warming up AI models...")
    await system_health_check()
    
    logger.success("✅ System ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    ngrok.kill()

# ============================================================================
# MAIN LAUNCHER
# ============================================================================

def main():
    """Main entry point for Colab"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║         🔷 ELITE MULTI-AGENT ORCHESTRATION SYSTEM 🔷                  ║
    ║                                                                       ║
    ║  Architecture: LangGraph + RAG + Qwen3-1.7B + FastAPI               ║
    ║  Agents: CodeArchitect | SecAnalyst | AutoBot | AgentSuite           ║
    ║          CreativeAgent | AgCustom                                    ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    port = 8000
    
    # Setup NGROK
    try:
        public_url = setup_ngrok(port)
        print(f"\n✅ PUBLIC URL: {public_url}")
        print(f"📊 Dashboard: {public_url}")
        print(f"📖 API Docs: {public_url}/api/docs")
        print(f"📈 Metrics: {public_url}/metrics\n")
    except Exception as e:
        logger.error(f"NGROK setup failed: {e}")
        logger.warning("Running without NGROK tunnel")
        public_url = None
    
    # Start server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        if public_url:
            ngrok.kill()

if __name__ == "__main__":
    main()
