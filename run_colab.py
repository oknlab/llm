"""
ELITE AI PLATFORM - COLAB RUNTIME
Production deployment with FastAPI + NGROK + Dashboard
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import time
import subprocess
import json

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Environment & Logging
from dotenv import load_dotenv
from loguru import logger

# System imports
from system import (
    get_orchestrator,
    get_metrics,
    Config,
    AgentOrchestrator
)

# NGROK
from pyngrok import ngrok, conf

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

logger.add(
    "logs/platform_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level=os.getenv("LOG_LEVEL", "INFO")
)

# ============================================
# PYDANTIC MODELS
# ============================================

class QueryRequest(BaseModel):
    """User query request"""
    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    agent: Optional[str] = Field(None, description="Specific agent to use")
    enable_rag: bool = Field(True, description="Enable RAG retrieval")


class CustomAgentRequest(BaseModel):
    """Custom agent creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    capabilities: list[str] = Field(..., min_items=1)
    system_prompt: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    """Query response"""
    status: str
    output: Optional[str] = None
    agents_used: Optional[list[str]] = None
    iterations: Optional[int] = None
    rag_context_used: Optional[bool] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("🚀 Starting Elite AI Platform...")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    logger.info("✅ Agent orchestrator initialized")
    
    # Start vLLM server
    vllm_process = start_vllm_server()
    
    # Setup NGROK
    ngrok_url = setup_ngrok()
    logger.info(f"🌐 Public URL: {ngrok_url}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down...")
    if vllm_process:
        vllm_process.terminate()
    ngrok.kill()


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Elite Multi-Agent AI Platform",
    description="Enterprise-grade custom AI agent creation platform",
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

# ============================================
# ROUTES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve main dashboard"""
    dashboard_path = Path("dashboard.html")
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    
    return HTMLResponse(dashboard_path.read_text())


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process user query through multi-agent system"""
    start_time = time.time()
    
    try:
        orchestrator = get_orchestrator()
        metrics = get_metrics()
        
        # Process query
        result = orchestrator.process(request.query)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        if result.get("agents_used"):
            for agent in result["agents_used"]:
                metrics.record_request(agent, processing_time)
        
        return QueryResponse(
            status=result["status"],
            output=result.get("output"),
            agents_used=result.get("agents_used"),
            iterations=result.get("iterations"),
            rag_context_used=result.get("rag_context_used"),
            error=result.get("error"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/create")
async def create_custom_agent(request: CustomAgentRequest):
    """Create custom AI agent"""
    try:
        orchestrator = get_orchestrator()
        
        result = orchestrator.create_custom_agent(
            name=request.name,
            description=request.description,
            capabilities=request.capabilities,
            system_prompt=request.system_prompt
        )
        
        if result["status"] == "success":
            return JSONResponse(content=result, status_code=201)
        else:
            raise HTTPException(status_code=400, detail=result.get("error"))
            
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def list_agents():
    """List all available agents"""
    try:
        orchestrator = get_orchestrator()
        agents = orchestrator.list_agents()
        return JSONResponse(content=agents)
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_system_metrics():
    """Get system metrics"""
    try:
        metrics = get_metrics()
        stats = metrics.get_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "metrics": stats,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "service": "Elite AI Platform",
        "version": "1.0.0"
    })


# ============================================
# VLLM SERVER MANAGEMENT
# ============================================

def start_vllm_server() -> Optional[subprocess.Popen]:
    """Start vLLM server for Qwen model"""
    try:
        logger.info("Starting vLLM server...")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", Config.MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--dtype", "auto",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.9"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(10)
        logger.info("✅ vLLM server started")
        
        return process
        
    except Exception as e:
        logger.warning(f"vLLM server start failed: {e}")
        logger.info("Using fallback LLM client")
        return None


# ============================================
# NGROK SETUP
# ============================================

def setup_ngrok() -> str:
    """Setup NGROK tunnel"""
    try:
        auth_token = os.getenv("NGROK_AUTH_TOKEN")
        if auth_token:
            conf.get_default().auth_token = auth_token
            conf.get_default().region = os.getenv("NGROK_REGION", "us")
        
        # Start tunnel
        port = int(os.getenv("API_PORT", "7860"))
        public_url = ngrok.connect(port, bind_tls=True)
        
        return public_url.public_url
        
    except Exception as e:
        logger.error(f"NGROK setup failed: {e}")
        return "http://localhost:7860"


# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point"""
    logger.info("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ELITE MULTI-AGENT AI PLATFORM                        ║
    ║     Production-Grade Custom Agent Creation               ║
    ║                                                          ║
    ║     • 6 Core Specialized Agents                          ║
    ║     • Custom Agent Factory                               ║
    ║     • Live Web Scraping RAG                              ║
    ║     • LangGraph Orchestration                            ║
    ║     • Enterprise Architecture                            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run FastAPI server
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "7860")),
        workers=int(os.getenv("API_WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()
