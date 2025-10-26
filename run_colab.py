"""
Elite AI System - Colab Launcher
FastAPI + NGROK + Multi-Agent RAG System
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Environment setup
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from pyngrok import ngrok, conf

# Import system
from system import get_orchestrator, AgentType, config

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Elite AI Agent System",
    description="Agentic RAG with Qwen3-1.7B",
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

# ═══════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    agents: Optional[List[str]] = Field(default=None, description="Agent types to use")
    use_rag: bool = Field(default=True, description="Enable RAG")

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    rag_enabled: bool

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend interface"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Elite AI System Active</h1>", status_code=200)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    orchestrator = get_orchestrator()
    return HealthResponse(
        status="operational",
        model=config.model_name,
        device=orchestrator.llm.device,
        rag_enabled=True
    )

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process query with multi-agent system"""
    try:
        orchestrator = get_orchestrator()
        
        # Parse agent types
        agent_types = []
        if request.agents:
            agent_map = {
                "CodeArchitect": AgentType.CODE_ARCHITECT,
                "SecAnalyst": AgentType.SEC_ANALYST,
                "AutoBot": AgentType.AUTO_BOT,
                "AgCustom": AgentType.AG_CUSTOM
            }
            agent_types = [agent_map[a] for a in request.agents if a in agent_map]
        else:
            agent_types = [AgentType.CODE_ARCHITECT]
        
        # Execute orchestration
        result = await orchestrator.orchestrate(
            query=request.query,
            agents=agent_types,
            use_rag=request.use_rag
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_and_index(request: Request):
    """Scrape URLs and index for RAG"""
    try:
        data = await request.json()
        query = data.get("query", "")
        max_pages = data.get("max_pages", 5)
        
        orchestrator = get_orchestrator()
        documents = orchestrator.rag.scraper.search_and_scrape(query, max_results=max_pages)
        
        if documents:
            orchestrator.rag.ingest_documents(documents)
        
        return JSONResponse(content={
            "status": "success",
            "documents_scraped": len(documents),
            "query": query
        })
    
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return JSONResponse(content={
        "agents": [
            {"name": "CodeArchitect", "description": "Engineering & code architecture"},
            {"name": "SecAnalyst", "description": "Security analysis & pen testing"},
            {"name": "AutoBot", "description": "Automation & API workflows"},
            {"name": "AgCustom", "description": "Custom AI agent builder"}
        ]
    })

# ═══════════════════════════════════════════════════════════
# NGROK TUNNEL
# ═══════════════════════════════════════════════════════════

def setup_ngrok():
    """Configure and start ngrok tunnel"""
    if not config.ngrok_auth_token or config.ngrok_auth_token == "your_ngrok_auth_token_here":
        logger.warning("NGROK_AUTH_TOKEN not set. Tunnel not created.")
        return None
    
    try:
        conf.get_default().auth_token = config.ngrok_auth_token
        public_url = ngrok.connect(config.port, bind_tls=True)
        logger.info(f"NGROK Tunnel: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"NGROK setup failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("=" * 60)
    logger.info("ELITE AI AGENT SYSTEM INITIALIZING")
    logger.info("=" * 60)
    
    # Initialize orchestrator (loads model)
    get_orchestrator()
    
    # Setup ngrok
    tunnel = setup_ngrok()
    if tunnel:
        logger.info(f"Public endpoint: {tunnel.public_url}")
    
    logger.info("System ready for requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down system...")
    ngrok.disconnect_all()

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Launch the system"""
    logger.info("Starting Elite AI System...")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
