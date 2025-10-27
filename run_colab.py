"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  ELITE RAG MULTI-AGENT SYSTEM - FASTAPI SERVER                            ║
║  Production-Grade API with Full Integration Dashboard                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from pyngrok import ngrok, conf

# Import system
from system import (
    get_agent_system,
    get_integration_hub,
    get_metrics,
    config,
    AgentType,
    AirflowDAGManager,
    logger
)

# ═══════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    agent: str = Field(default="CodeArchitect")
    enable_web_search: bool = Field(default=True)

class IntegrationRequest(BaseModel):
    integration_type: str = Field(..., description="slack|n8n|zapier|manufacturing|erp")
    data: dict

class DAGRequest(BaseModel):
    dag_type: str = Field(..., description="rag_sync|manufacturing|custom")

# ═══════════════════════════════════════════════════════════
# LIFESPAN MANAGEMENT
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("🚀 STARTING ELITE RAG MULTI-AGENT SYSTEM")
    logger.info("=" * 70)
    
    # Initialize system (loads model)
    agent_system = get_agent_system()
    logger.info(f"✓ Agent System initialized with {len(AgentType)} agents")
    
    # Setup NGROK tunnel
    if config.ngrok_auth_token and config.ngrok_auth_token != "your_ngrok_token_here":
        try:
            conf.get_default().auth_token = config.ngrok_auth_token
            public_url = ngrok.connect(config.port, bind_tls=True)
            logger.info(f"🌐 NGROK Tunnel: {public_url}")
            logger.info(f"📊 Dashboard: {public_url}/dashboard")
        except Exception as e:
            logger.error(f"NGROK failed: {e}")
    
    logger.info("=" * 70)
    logger.info("✓ SYSTEM READY FOR REQUESTS")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("Shutting down system...")
    ngrok.disconnect_all()

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Elite RAG Multi-Agent System",
    description="Production-grade agentic RAG with LangGraph, Airflow, and full integrations",
    version="2.0.0",
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

# ═══════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard"""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/dashboard">
        </head>
        <body>
            <p>Redirecting to dashboard...</p>
        </body>
    </html>
    """

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve main dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

@app.get("/health")
async def health():
    """Health check"""
    agent_system = get_agent_system()
    return {
        "status": "operational",
        "model": config.model_name,
        "device": agent_system.llm.device,
        "vector_db_docs": agent_system.rag.vectorstore._collection.count(),
        "agents_available": [a.value for a in AgentType],
        "integrations": {
            "slack": bool(config.slack_bot_token),
            "notion": bool(config.notion_token),
            "github": bool(config.github_token),
            "manufacturing": bool(config.printer_3d_api_url),
            "erp": bool(config.erp_system_url)
        }
    }

@app.post("/query")
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Execute agent query"""
    try:
        agent_system = get_agent_system()
        
        # Validate agent type
        if request.agent not in [a.value for a in AgentType]:
            raise HTTPException(status_code=400, detail=f"Invalid agent: {request.agent}")
        
        # Execute workflow
        result = await agent_system.run(request.query, request.agent)
        
        # Log to MLflow in background
        if config.mlflow_tracking_uri:
            background_tasks.add_task(log_to_mlflow, result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integration/trigger")
async def trigger_integration(request: IntegrationRequest):
    """Trigger external integration"""
    try:
        hub = get_integration_hub()
        
        result = None
        if request.integration_type == "slack":
            hub.send_slack_message(
                channel=request.data.get('channel', '#general'),
                message=request.data.get('message', '')
            )
            result = {"status": "sent"}
            
        elif request.integration_type == "n8n":
            hub.trigger_n8n_workflow(request.data)
            result = {"status": "triggered"}
            
        elif request.integration_type == "zapier":
            hub.trigger_zapier_zap(request.data)
            result = {"status": "triggered"}
            
        elif request.integration_type == "manufacturing":
            result = hub.send_to_manufacturing(request.data)
            
        elif request.integration_type == "erp":
            result = hub.update_erp_system(request.data)
        
        else:
            raise HTTPException(status_code=400, detail="Unknown integration type")
        
        return {"integration": request.integration_type, "result": result}
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/airflow/dags")
async def list_dags():
    """List available Airflow DAGs"""
    return {
        "dags": [
            {"name": "rag_sync", "description": "RAG knowledge base sync"},
            {"name": "manufacturing", "description": "Manufacturing pipeline"},
        ]
    }

@app.post("/airflow/dag/generate")
async def generate_dag(request: DAGRequest):
    """Generate Airflow DAG code"""
    try:
        manager = AirflowDAGManager()
        
        if request.dag_type == "rag_sync":
            dag_code = manager.generate_rag_sync_dag()
        elif request.dag_type == "manufacturing":
            dag_code = manager.generate_manufacturing_dag()
        else:
            raise HTTPException(status_code=400, detail="Unknown DAG type")
        
        return PlainTextResponse(content=dag_code, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"DAG generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(content=get_metrics(), media_type="text/plain")

@app.get("/agents")
async def list_agents():
    """List available agents"""
    agents = []
    for agent in AgentType:
        agents.append({
            "name": agent.value,
            "description": {
                "CodeArchitect": "Software engineering & system design",
                "SecAnalyst": "Security analysis & penetration testing",
                "AutoBot": "Automation & API workflows",
                "AgentSuite": "Administrative & operational management",
                "CreativeAgent": "Content creation & creative writing",
                "AgCustom": "Custom AI agent solutions"
            }[agent.value]
        })
    return {"agents": agents}

@app.post("/web-scrape")
async def web_scrape(query: str, max_results: int = 10):
    """Manually trigger web scraping"""
    try:
        agent_system = get_agent_system()
        documents = agent_system.rag.ingest_web_content(query)
        
        return {
            "query": query,
            "documents_scraped": len(documents),
            "sources": [doc.metadata['source'] for doc in documents[:5]]
        }
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-db/stats")
async def vector_db_stats():
    """Vector database statistics"""
    agent_system = get_agent_system()
    return {
        "total_documents": agent_system.rag.vectorstore._collection.count(),
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "top_k": config.top_k
    }

# ═══════════════════════════════════════════════════════════
# BACKGROUND TASKS
# ═══════════════════════════════════════════════════════════

def log_to_mlflow(result: dict):
    """Log query results to MLflow"""
    try:
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        with mlflow.start_run():
            mlflow.log_param("agent", result['agent'])
            mlflow.log_param("query", result['query'][:100])
            mlflow.log_metric("confidence", result['confidence'])
            mlflow.log_metric("sources_used", len(result['sources']))
            mlflow.log_metric("iterations", result['iterations'])
        
        logger.info("Logged to MLflow")
    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    """Launch the system"""
    logger.info("Starting FastAPI server...")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()
