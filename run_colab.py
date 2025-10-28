"""
ENTERPRISE MULTI-AGENT SYSTEM - COLAB RUNNER
Complete orchestration: vLLM server, NGROK tunnel, FastAPI, Dashboard
"""

import os
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Optional
import signal
import sys

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger
from pyngrok import ngrok
import uvicorn

from system import (
    AgentOrchestrator,
    LiveRAGEngine,
    SystemAPI,
    TaskState
)
from langchain_community.chat_models import ChatOpenAI


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

class Config:
    """System configuration"""
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
    MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen3-1.7B')
    MODEL_SERVER = os.getenv('MODEL_SERVER', 'http://localhost:8000/v1')
    VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'sk-vllm-internal-2024')
    GOOGLE_CSE_URL = os.getenv('GOOGLE_CSE_URL')
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8080))
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8090))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', 8192))


config = Config()

# Configure logging
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL)
logger.add("logs/system_{time}.log", rotation="500 MB", retention="10 days")


# ============================================================================
# VLLM SERVER MANAGEMENT
# ============================================================================

class VLLMManager:
    """Manage vLLM server lifecycle"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.server_url = config.MODEL_SERVER
    
    def start_server(self):
        """Start vLLM server"""
        logger.info(f"Starting vLLM server with {config.MODEL_NAME}")
        
        cmd = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', config.MODEL_NAME,
            '--host', '0.0.0.0',
            '--port', '8000',
            '--max-model-len', str(config.CONTEXT_WINDOW),
            '--dtype', 'half',
            '--gpu-memory-utilization', '0.9',
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server ready
        logger.info("Waiting for vLLM server to be ready...")
        time.sleep(30)  # Model loading time
        logger.info(f"vLLM server running at {self.server_url}")
    
    def stop_server(self):
        """Stop vLLM server"""
        if self.process:
            logger.info("Stopping vLLM server")
            self.process.terminate()
            self.process.wait(timeout=10)
            logger.info("vLLM server stopped")


# ============================================================================
# NGROK TUNNEL
# ============================================================================

class NGROKManager:
    """Manage NGROK tunnels"""
    
    def __init__(self):
        self.tunnels = {}
    
    def start_tunnel(self, port: int, name: str) -> str:
        """Start NGROK tunnel"""
        if config.NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
        
        tunnel = ngrok.connect(port, bind_tls=True)
        public_url = tunnel.public_url
        self.tunnels[name] = tunnel
        
        logger.info(f"NGROK tunnel [{name}] -> {public_url}")
        return public_url
    
    def stop_all(self):
        """Stop all tunnels"""
        ngrok.kill()
        logger.info("All NGROK tunnels closed")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Enterprise Multi-Agent System",
    description="Production-grade AI agent orchestration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vllm_manager: Optional[VLLMManager] = None
ngrok_manager: Optional[NGROKManager] = None
system_api: Optional[SystemAPI] = None


# ============================================================================
# API MODELS
# ============================================================================

class TaskRequest(BaseModel):
    objective: str
    metadata: dict = {}


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global vllm_manager, ngrok_manager, system_api
    
    logger.info("=" * 80)
    logger.info("ENTERPRISE MULTI-AGENT SYSTEM - STARTUP")
    logger.info("=" * 80)
    
    # Start vLLM server
    vllm_manager = VLLMManager()
    vllm_manager.start_server()
    
    # Initialize LLM client
    llm = ChatOpenAI(
        base_url=config.MODEL_SERVER,
        api_key=config.VLLM_API_KEY,
        model=config.MODEL_NAME,
        temperature=0.7,
        max_tokens=2048
    )
    
    # Initialize RAG engine
    rag_engine = LiveRAGEngine(config.GOOGLE_CSE_URL)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(llm, rag_engine)
    
    # Initialize API layer
    system_api = SystemAPI(orchestrator)
    
    # Start NGROK tunnels
    ngrok_manager = NGROKManager()
    api_url = ngrok_manager.start_tunnel(config.API_PORT, "api")
    dashboard_url = ngrok_manager.start_tunnel(config.DASHBOARD_PORT, "dashboard")
    
    logger.info(f"API accessible at: {api_url}")
    logger.info(f"Dashboard accessible at: {dashboard_url}")
    logger.info("System ready for tasks")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down system...")
    if vllm_manager:
        vllm_manager.stop_server()
    if ngrok_manager:
        ngrok_manager.stop_all()
    logger.info("Shutdown complete")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system info"""
    return """
    <html>
        <head><title>Enterprise Multi-Agent System</title></head>
        <body style="font-family: monospace; padding: 20px;">
            <h1>🤖 Enterprise Multi-Agent System</h1>
            <h2>Endpoints:</h2>
            <ul>
                <li><a href="/docs">/docs</a> - API Documentation</li>
                <li><a href="/health">/health</a> - Health Check</li>
                <li><a href="/tasks">/tasks</a> - List Tasks</li>
                <li>POST /task - Submit Task</li>
                <li>GET /task/{task_id} - Task Status</li>
            </ul>
            <h2>Agents:</h2>
            <p>CodeArchitect | SecAnalyst | AutoBot | AgentSuite | CreativeAgent | AgCustom</p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": config.MODEL_NAME,
        "model_server": config.MODEL_SERVER,
        "agents": ["CodeArchitect", "SecAnalyst", "AutoBot", "AgentSuite", "CreativeAgent", "AgCustom"]
    }


@app.post("/task", response_model=TaskResponse)
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Submit new task to orchestrator"""
    if not system_api:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await system_api.submit_task(request.objective)
        
        return TaskResponse(
            task_id=result['task_id'],
            status=result['status'],
            message=f"Task submitted. Agents: {', '.join(result['agents_executed'])}"
        )
    except Exception as e:
        logger.error(f"Task submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and results"""
    if not system_api:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    result = system_api.get_task_status(task_id)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return result


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    if not system_api:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {"tasks": system_api.list_tasks()}


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve dashboard HTML"""
    dashboard_path = Path("dashboard.html")
    if dashboard_path.exists():
        return dashboard_path.read_text()
    else:
        return "<h1>Dashboard not found</h1>"


# ============================================================================
# MAIN RUNNER
# ============================================================================

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    sys.exit(0)


def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Enterprise Multi-Agent System...")
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
