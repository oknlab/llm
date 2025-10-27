"""
Colab Orchestration Runner
Handles model server, ngrok tunneling, FastAPI, monitoring
"""

import os
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

# FastAPI
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Server
import uvicorn

# Ngrok
from pyngrok import ngrok, conf

# Monitoring
from prometheus_client import make_asgi_app, generate_latest

# Logging
from loguru import logger

# System
from system import (
    get_system, SystemManager, Task, AgentType, TaskPriority,
    AGENT_REQUESTS, AGENT_LATENCY, ACTIVE_AGENTS
)

# Config
from dotenv import load_dotenv
load_dotenv()


# ==================== CONFIGURATION ====================
class Config:
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    PUBLIC_PORT = int(os.getenv("PUBLIC_PORT", 7860))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ==================== LOGGING SETUP ====================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=Config.LOG_LEVEL
)
logger.add(
    "logs/system_{time:YYYY-MM-DD}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)


# ==================== LIFESPAN MANAGER ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("🚀 Starting AI Agent Orchestration System")
    
    # Setup ngrok
    if Config.NGROK_AUTH_TOKEN:
        conf.get_default().auth_token = Config.NGROK_AUTH_TOKEN
        tunnel = ngrok.connect(Config.PUBLIC_PORT, bind_tls=True)
        public_url = tunnel.public_url
        logger.info(f"🌐 Ngrok tunnel: {public_url}")
        app.state.public_url = public_url
    else:
        logger.warning("⚠️  No NGROK_AUTH_TOKEN provided")
        app.state.public_url = f"http://localhost:{Config.PUBLIC_PORT}"
    
    # Initialize system
    app.state.system = get_system()
    logger.info("✅ System initialized")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down...")
    if Config.NGROK_AUTH_TOKEN:
        ngrok.disconnect(tunnel.public_url)
    await app.state.system.shutdown()


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="AI Agent Orchestration System",
    description="Enterprise Multi-Agent Platform with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ==================== WEBSOCKET MANAGER ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard"""
    with open("dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    system: SystemManager = app.state.system
    metrics = system.get_system_metrics()
    return JSONResponse({
        "status": "healthy",
        "environment": Config.ENVIRONMENT,
        "public_url": app.state.public_url,
        "metrics": metrics
    })


@app.post("/api/v1/task")
async def submit_task(task_data: Dict[str, Any]):
    """Submit task to agent system"""
    try:
        task = Task(
            description=task_data.get("description", ""),
            agent_type=AgentType(task_data["agent_type"]) if task_data.get("agent_type") else None,
            priority=TaskPriority(task_data.get("priority", "medium")),
            context=task_data.get("context", {})
        )
        
        system: SystemManager = app.state.system
        result = await system.submit_task(task)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "task_completed",
            "data": result
        })
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Task submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents")
async def list_agents():
    """List available agents"""
    return JSONResponse({
        "agents": [
            {
                "type": agent.value,
                "description": agent.name
            }
            for agent in AgentType
        ]
    })


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics"""
    system: SystemManager = app.state.system
    return JSONResponse(system.get_system_metrics())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo for heartbeat
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/v1/rag/build")
async def build_knowledge_base(data: Dict[str, Any]):
    """Build RAG knowledge base from query"""
    try:
        system: SystemManager = app.state.system
        query = data.get("query", "")
        
        await system.orchestrator.rag.build_knowledge_base(query)
        
        return JSONResponse({
            "status": "success",
            "message": f"Knowledge base built for: {query}"
        })
    except Exception as e:
        logger.error(f"RAG build error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SIGNAL HANDLERS ====================
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== MAIN ====================
def main():
    """Run the server"""
    logger.info(f"🔥 Starting server on port {Config.PUBLIC_PORT}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PUBLIC_PORT,
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True,
        workers=1  # Single worker for Colab
    )


if __name__ == "__main__":
    main()
