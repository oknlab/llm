"""
Google Colab Entry Point - AI Agent Platform
Handles vLLM server setup, NGROK tunneling, FastAPI server
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import nest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn

# Apply nest_asyncio for Colab compatibility
nest_asyncio.apply()

# Import system
from system import (
    AIAgentPlatform,
    AgentRole,
    TaskStatus,
    Config,
    get_platform
)

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

# NGROK Configuration
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
API_PORT = int(os.getenv("API_PORT", "7860"))

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AI Agent Platform API",
    description="Enterprise Multi-Agent Orchestration System",
    version="1.0.0"
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
# REQUEST/RESPONSE MODELS
# ============================================

class TaskCreateRequest(BaseModel):
    task_type: str
    description: str
    parameters: dict = {}
    use_rag: bool = False
    rag_query: Optional[str] = None


class TaskExecuteRequest(BaseModel):
    task_id: str


class CustomAgentRequest(BaseModel):
    agent_id: str
    system_prompt: str
    capabilities: list[str]


class QueryRequest(BaseModel):
    query: str
    agent_role: Optional[str] = None
    use_rag: bool = False
    rag_query: Optional[str] = None


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return dashboard_path.read_text()
    return "<h1>AI Agent Platform API</h1><p>Dashboard not found. Access /docs for API documentation.</p>"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@app.post("/api/tasks/create")
async def create_task(request: TaskCreateRequest):
    """Create new task"""
    try:
        platform = get_platform()
        
        # Add RAG parameters
        params = request.parameters.copy()
        params["use_rag"] = request.use_rag
        params["rag_query"] = request.rag_query
        
        task = await platform.create_task(
            task_type=request.task_type,
            description=request.description,
            parameters=params
        )
        
        return {
            "success": True,
            "task_id": task.task_id,
            "status": task.status.value
        }
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tasks/execute")
async def execute_task(request: TaskExecuteRequest, background_tasks: BackgroundTasks):
    """Execute task by ID"""
    try:
        platform = get_platform()
        
        # Execute in background
        background_tasks.add_task(platform.execute_task, request.task_id)
        
        return {
            "success": True,
            "message": f"Task {request.task_id} execution started",
            "task_id": request.task_id
        }
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and result"""
    try:
        platform = get_platform()
        task = platform.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks")
async def list_tasks():
    """List all tasks"""
    try:
        platform = get_platform()
        tasks = platform.list_tasks()
        
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "task_type": t.task_type,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat()
                }
                for t in tasks
            ]
        }
    except Exception as e:
        logger.error(f"List tasks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_agent(request: QueryRequest):
    """Direct query to agent system"""
    try:
        platform = get_platform()
        
        result = await platform.orchestrator.execute_task(
            task=request.query,
            context={"agent_role": request.agent_role},
            use_rag=request.use_rag,
            rag_query=request.rag_query
        )
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/custom/create")
async def create_custom_agent(request: CustomAgentRequest):
    """Create custom agent"""
    try:
        platform = get_platform()
        
        agent = platform.orchestrator.custom_builder.create_custom_agent(
            agent_id=request.agent_id,
            system_prompt=request.system_prompt,
            capabilities=request.capabilities
        )
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "message": "Custom agent created successfully"
        }
    except Exception as e:
        logger.error(f"Custom agent creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {"role": "code_architect", "description": "Code architecture and development"},
            {"role": "sec_analyst", "description": "Security analysis and pentesting"},
            {"role": "auto_bot", "description": "Automation and workflows"},
            {"role": "creative_agent", "description": "Creative content generation"},
            {"role": "agent_suite", "description": "Business operations and admin"},
        ]
    }


# ============================================
# VLLM SERVER MANAGEMENT
# ============================================

class VLLMServer:
    """vLLM server manager"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.model_name = Config.MODEL_NAME
        self.port = int(os.getenv("VLLM_PORT", "8000"))
    
    def start(self):
        """Start vLLM server"""
        logger.info(f"Starting vLLM server with model: {self.model_name}")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--max-model-len", os.getenv("VLLM_MAX_MODEL_LEN", "4096"),
            "--gpu-memory-utilization", os.getenv("VLLM_GPU_MEMORY", "0.90"),
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            logger.info("Waiting for vLLM server to start...")
            time.sleep(30)
            
            logger.info(f"vLLM server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            raise
    
    def stop(self):
        """Stop vLLM server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("vLLM server stopped")


# ============================================
# NGROK TUNNEL
# ============================================

def setup_ngrok():
    """Setup NGROK tunnel"""
    if not NGROK_AUTH_TOKEN:
        logger.warning("NGROK_AUTH_TOKEN not set, skipping tunnel setup")
        return None
    
    try:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        tunnel = ngrok.connect(API_PORT, bind_tls=True)
        public_url = tunnel.public_url
        
        logger.info(f"NGROK tunnel established: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"NGROK setup failed: {e}")
        return None


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("AI AGENT PLATFORM - STARTING")
    logger.info("=" * 60)
    
    # Setup directories
    Config.setup_directories()
    
    # Start vLLM server
    vllm_server = VLLMServer()
    
    try:
        logger.info("Step 1: Starting vLLM model server...")
        vllm_server.start()
        
        logger.info("Step 2: Setting up NGROK tunnel...")
        public_url = setup_ngrok()
        
        if public_url:
            logger.info(f"Public URL: {public_url}")
            logger.info(f"Dashboard: {public_url}")
            logger.info(f"API Docs: {public_url}/docs")
        
        logger.info("Step 3: Starting FastAPI server...")
        logger.info(f"Local URL: http://localhost:{API_PORT}")
        
        # Run FastAPI server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=API_PORT,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        vllm_server.stop()
        ngrok.kill()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        logger.info("Running in Google Colab environment")
    except:
        IN_COLAB = False
        logger.info("Running in local environment")
    
    main()
