"""
ELITE SELF-IMPROVING MULTI-AGENT ECOSYSTEM - LAUNCHER
LangGraph + Airflow + Multi-Source RAG + Integrations
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from pyngrok import ngrok, conf
from prometheus_fastapi_instrumentator import Instrumentator

from system import (
    get_system, get_improvement_engine,
    AgentType, FeedbackData, config, METRICS
)

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Elite Multi-Agent Ecosystem",
    description="LangGraph + Airflow + Multi-Source RAG",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# ═══════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    agents: Optional[List[str]] = None
    use_rag: bool = True
    sources: Optional[List[str]] = ['google_cse']

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: float
    agent: str

class IntegrationRequest(BaseModel):
    action: str  # 'slack', 'github', 'notion', 'webhook'
    params: dict

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Elite Multi-Agent Ecosystem</title>
        <style>
            body { font-family: Arial; background: #0a0e27; color: #e0e6ff; padding: 50px; }
            h1 { color: #00d4ff; }
            .status { background: #1a1f3a; padding: 20px; border-radius: 10px; }
            .metric { margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🔷 Elite Multi-Agent Ecosystem - v2.0</h1>
        <div class="status">
            <h2>System Status: OPERATIONAL</h2>
            <div class="metric">✅ LangGraph Agents Active</div>
            <div class="metric">✅ Multi-Source RAG Online</div>
            <div class="metric">✅ Airflow DAGs Scheduled</div>
            <div class="metric">✅ Integrations Ready (Slack, GitHub, Notion, etc.)</div>
            <div class="metric">✅ Monitoring Active (Prometheus, Grafana)</div>
            <div class="metric">✅ Self-Improvement Enabled</div>
        </div>
        <p><a href="/docs" style="color: #00d4ff;">API Documentation</a></p>
        <p><a href="/metrics" style="color: #00d4ff;">Prometheus Metrics</a></p>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    system = get_system()
    return {
        "status": "operational",
        "model": config.model_name,
        "device": system.llm.device,
        "integrations": {
            "slack": bool(config.slack_bot_token),
            "github": bool(config.github_token),
            "notion": bool(config.notion_api_key),
            "kafka": bool(config.kafka_bootstrap_servers),
            "wandb": bool(config.wandb_api_key),
        },
        "self_improvement": config.enable_self_improvement
    }

@app.post("/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Execute multi-agent query"""
    try:
        system = get_system()
        
        # Parse agents
        agent_map = {
            "CodeArchitect": AgentType.CODE_ARCHITECT,
            "SecAnalyst": AgentType.SEC_ANALYST,
            "AutoBot": AgentType.AUTO_BOT,
            "DataEngineer": AgentType.DATA_ENGINEER,
            "MLOpsAgent": AgentType.MLOPS_AGENT,
        }
        
        agents = [agent_map[a] for a in (request.agents or ["CodeArchitect"]) if a in agent_map]
        
        # Ingest from sources in background
        if request.use_rag and request.sources:
            background_tasks.add_task(
                system.rag.ingest_from_sources,
                request.query,
                request.sources
            )
        
        # Execute orchestration
        result = await system.orchestrate(
            query=request.query,
            agents=agents,
            use_rag=request.use_rag
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for self-improvement"""
    try:
        engine = get_improvement_engine()
        feedback = FeedbackData(
            query=request.query,
            response=request.response,
            rating=request.rating,
            agent=request.agent,
            timestamp=datetime.utcnow().isoformat()
        )
        engine.record_feedback(feedback)
        
        return {"status": "recorded", "rating": request.rating}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations")
async def trigger_integration(request: IntegrationRequest):
    """Trigger external integrations"""
    try:
        system = get_system()
        
        if request.action == "slack":
            await system.integrations.send_slack(
                request.params.get("message"),
                request.params.get("channel")
            )
        
        elif request.action == "github":
            docs = await system.integrations.fetch_github_repo(
                request.params.get("repo")
            )
            system.rag.ingest_documents(docs)
        
        elif request.action == "notion":
            docs = await system.integrations.query_notion(
                request.params.get("database_id")
            )
            # Process Notion docs
        
        elif request.action == "webhook":
            await system.integrations.trigger_webhook(
                request.params.get("url"),
                request.params.get("data")
            )
        
        return {"status": "executed", "action": request.action}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "CodeArchitect", "type": "engineering"},
            {"name": "SecAnalyst", "type": "security"},
            {"name": "AutoBot", "type": "automation"},
            {"name": "DataEngineer", "type": "data_pipelines"},
            {"name": "MLOpsAgent", "type": "mlops"},
        ]
    }

@app.post("/scrape")
async def scrape_sources(background_tasks: BackgroundTasks, query: str, sources: List[str]):
    """Scrape and index from multiple sources"""
    system = get_system()
    background_tasks.add_task(
        system.rag.ingest_from_sources,
        query,
        sources
    )
    return {"status": "scraping_started", "query": query, "sources": sources}

# ═══════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    logger.info("=" * 70)
    logger.info("ELITE SELF-IMPROVING MULTI-AGENT ECOSYSTEM v2.0")
    logger.info("=" * 70)
    
    # Initialize system
    get_system()
    
    # Start Prometheus metrics server
    # start_http_server(config.prometheus_port)
    
    # Setup NGROK
    if config.ngrok_auth_token and config.ngrok_auth_token != "your_ngrok_token_here":
        conf.get_default().auth_token = config.ngrok_auth_token
        tunnel = ngrok.connect(config.port, bind_tls=True)
        logger.info(f"🌐 NGROK Tunnel: {tunnel.public_url}")
    
    logger.info("✅ System Online")

@app.on_event("shutdown")
async def shutdown():
    ngrok.disconnect_all()
    logger.info("System shutdown complete")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("Launching Elite Multi-Agent Ecosystem...")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    main()
