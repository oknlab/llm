"""
ELITE MULTI-AGENT SYSTEM - PRODUCTION RUNNER
FastAPI + Airflow + NGROK Integration
Execute in Google Colab or Local Environment
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import json
from pathlib import Path
import uuid

# Environment
from dotenv import load_dotenv
load_dotenv()

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# NGROK
from pyngrok import ngrok, conf

# Monitoring
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil

# System
from system import AgentOrchestrator, AgentType, Task, TaskStatus

# ==============================================
# LOGGING
# ==============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_system.log')
    ]
)
logger = logging.getLogger(__name__)


# ==============================================
# CONFIGURATION
# ==============================================
class SystemConfig:
    """System-wide configuration"""
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # NGROK
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', '')
    NGROK_PORT = int(os.getenv('NGROK_PORT', 8000))
    
    # Model
    MODEL_CONFIG = {
        'MODEL_NAME': os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct'),
        'MODEL_DEVICE': os.getenv('MODEL_DEVICE', 'cuda'),
        'MODEL_QUANTIZATION': os.getenv('MODEL_QUANTIZATION', '8bit'),
        'MAX_NEW_TOKENS': os.getenv('MAX_NEW_TOKENS', '2048'),
        'TEMPERATURE': os.getenv('TEMPERATURE', '0.7'),
        'TOP_P': os.getenv('TOP_P', '0.95')
    }
    
    # RAG
    RAG_CONFIG = {
        'GOOGLE_CSE_ID': os.getenv('GOOGLE_CSE_ID', ''),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'CHUNK_SIZE': os.getenv('CHUNK_SIZE', '1000'),
        'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP', '200'),
        'TOP_K_RESULTS': os.getenv('TOP_K_RESULTS', '5')
    }
    
    # API
    API_KEY = os.getenv('API_KEY', 'sk-elite-agent-system-2024')
    CORS_ORIGINS = json.loads(os.getenv('CORS_ORIGINS', '["*"]'))


# ==============================================
# METRICS
# ==============================================
class Metrics:
    """Prometheus metrics"""
    
    task_counter = Counter('agent_tasks_total', 'Total tasks processed', ['agent', 'status'])
    task_duration = Histogram('agent_task_duration_seconds', 'Task duration', ['agent'])
    system_cpu = Counter('system_cpu_percent', 'CPU usage')
    system_memory = Counter('system_memory_percent', 'Memory usage')


# ==============================================
# GLOBAL STATE
# ==============================================
class GlobalState:
    """Application state"""
    orchestrator: Optional[AgentOrchestrator] = None
    tasks: Dict[str, Task] = {}
    metrics = Metrics()
    public_url: Optional[str] = None


# ==============================================
# PYDANTIC MODELS
# ==============================================
class TaskRequest(BaseModel):
    agent: str = Field(..., description="Agent type")
    query: str = Field(..., description="Task query", min_length=1)
    use_rag: bool = Field(True, description="Enable RAG")
    context: Dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskResult(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    agents: List[str]
    tasks_total: int
    tasks_pending: int
    tasks_running: int
    tasks_completed: int
    public_url: Optional[str]
    cpu_percent: float
    memory_percent: float
    uptime_seconds: float


# ==============================================
# LIFESPAN MANAGEMENT
# ==============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    logger.info("🚀 Starting Elite Multi-Agent System...")
    
    # Initialize orchestrator
    config = {**SystemConfig.MODEL_CONFIG, **SystemConfig.RAG_CONFIG}
    GlobalState.orchestrator = AgentOrchestrator(config)
    logger.info("✅ Agent Orchestrator initialized")
    
    # Setup NGROK tunnel
    if SystemConfig.NGROK_AUTH_TOKEN:
        try:
            conf.get_default().auth_token = SystemConfig.NGROK_AUTH_TOKEN
            tunnel = ngrok.connect(
                SystemConfig.NGROK_PORT,
                bind_tls=True
            )
            GlobalState.public_url = tunnel.public_url
            logger.info(f"🌐 NGROK Tunnel: {GlobalState.public_url}")
        except Exception as e:
            logger.error(f"NGROK setup failed: {e}")
    
    # Start time
    GlobalState.start_time = datetime.now()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    if SystemConfig.NGROK_AUTH_TOKEN:
        ngrok.kill()


# ==============================================
# FASTAPI APPLICATION
# ==============================================
app = FastAPI(
    title="Elite Multi-Agent Orchestration System",
    description="Enterprise-grade AI agent orchestration with RAG, LangGraph, and Airflow",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=SystemConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================
# BACKGROUND TASK PROCESSOR
# ==============================================
async def process_task_background(task_id: str, use_rag: bool):
    """Background task processor"""
    task = GlobalState.tasks.get(task_id)
    if not task:
        return
    
    try:
        task.status = TaskStatus.RUNNING
        start_time = datetime.now()
        
        # Process with orchestrator
        result = GlobalState.orchestrator.process_task(task, use_rag=use_rag)
        
        # Update task
        task.status = TaskStatus.COMPLETED if result['status'] == 'success' else TaskStatus.FAILED
        task.result = result.get('result')
        task.metadata = result.get('metadata', {})
        task.completed_at = datetime.now()
        
        # Metrics
        duration = (datetime.now() - start_time).total_seconds()
        Metrics.task_counter.labels(agent=task.agent_type.value, status=task.status.value).inc()
        Metrics.task_duration.labels(agent=task.agent_type.value).observe(duration)
        
        logger.info(f"✅ Task {task_id} completed in {duration:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}")
        task.status = TaskStatus.FAILED
        task.metadata['error'] = str(e)
        Metrics.task_counter.labels(agent=task.agent_type.value, status='failed').inc()


# ==============================================
# API ENDPOINTS
# ==============================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve dashboard"""
    dashboard_path = Path("dashboard.html")
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Elite Agent System</h1><p>Dashboard not found</p>")


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus)
async def status():
    """System status"""
    tasks = list(GlobalState.tasks.values())
    
    return SystemStatus(
        status="operational",
        agents=[agent.value for agent in AgentType if agent != AgentType.ORCHESTRATOR],
        tasks_total=len(tasks),
        tasks_pending=sum(1 for t in tasks if t.status == TaskStatus.PENDING),
        tasks_running=sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
        tasks_completed=sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
        public_url=GlobalState.public_url,
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        uptime_seconds=(datetime.now() - GlobalState.start_time).total_seconds()
    )


@app.post("/task", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and execute task"""
    try:
        # Validate agent
        try:
            agent_type = AgentType[request.agent.upper().replace('-', '_')]
        except KeyError:
            raise HTTPException(400, f"Invalid agent: {request.agent}")
        
        # Create task
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            agent_type=agent_type,
            query=request.query,
            context=request.context
        )
        
        GlobalState.tasks[task_id] = task
        
        # Schedule background processing
        background_tasks.add_task(process_task_background, task_id, request.use_rag)
        
        logger.info(f"📝 Task {task_id} created for agent {agent_type.value}")
        
        return TaskResponse(
            task_id=task_id,
            status="accepted",
            message=f"Task submitted to {agent_type.value}"
        )
    
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(500, str(e))


@app.get("/task/{task_id}", response_model=TaskResult)
async def get_task(task_id: str):
    """Get task result"""
    task = GlobalState.tasks.get(task_id)
    
    if not task:
        raise HTTPException(404, "Task not found")
    
    return TaskResult(
        task_id=task.id,
        status=task.status.value,
        result=task.result,
        metadata=task.metadata,
        error=task.metadata.get('error')
    )


@app.get("/tasks")
async def list_tasks(limit: int = 100, status: Optional[str] = None):
    """List tasks"""
    tasks = list(GlobalState.tasks.values())
    
    if status:
        try:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]
        except ValueError:
            pass
    
    tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]
    
    return {
        "total": len(tasks),
        "tasks": [
            {
                "id": t.id,
                "agent": t.agent_type.value,
                "status": t.status.value,
                "query": t.query[:100],
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None
            }
            for t in tasks
        ]
    }


@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {
                "type": agent.value,
                "name": agent.name,
                "description": f"Specialized {agent.value} agent"
            }
            for agent in AgentType
            if agent != AgentType.ORCHESTRATOR
        ]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    # Update system metrics
    Metrics.system_cpu.inc(psutil.cpu_percent())
    Metrics.system_memory.inc(psutil.virtual_memory().percent)
    
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ==============================================
# AIRFLOW DAGS
# ==============================================
def create_airflow_dags():
    """Create Airflow DAGs for scheduled tasks"""
    
    # DAG 1: System Health Check
    with DAG(
        'system_health_check',
        default_args={'owner': 'elite-system'},
        description='Monitor system health',
        schedule_interval='*/5 * * * *',  # Every 5 minutes
        start_date=days_ago(1),
        catchup=False
    ) as health_dag:
        
        def check_system_health():
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            logger.info(f"Health Check - CPU: {cpu}%, Memory: {memory}%")
            
            if cpu > 90 or memory > 90:
                logger.warning("System resources high!")
        
        health_task = PythonOperator(
            task_id='check_health',
            python_callable=check_system_health
        )
    
    # DAG 2: Periodic RAG Index Update
    with DAG(
        'rag_index_update',
        default_args={'owner': 'elite-system'},
        description='Update RAG indices',
        schedule_interval='0 */6 * * *',  # Every 6 hours
        start_date=days_ago(1),
        catchup=False
    ) as rag_dag:
        
        def update_rag_indices():
            logger.info("Updating RAG indices...")
            # Trigger index rebuild for common queries
            common_queries = [
                "latest technology trends",
                "software engineering best practices",
                "cybersecurity updates"
            ]
            
            for query in common_queries:
                if GlobalState.orchestrator:
                    GlobalState.orchestrator.rag.build_rag_index(query)
        
        update_task = PythonOperator(
            task_id='update_indices',
            python_callable=update_rag_indices
        )
    
    # DAG 3: Task Cleanup
    with DAG(
        'task_cleanup',
        default_args={'owner': 'elite-system'},
        description='Clean old tasks',
        schedule_interval='0 0 * * *',  # Daily
        start_date=days_ago(1),
        catchup=False
    ) as cleanup_dag:
        
        def cleanup_old_tasks():
            logger.info("Cleaning old tasks...")
            cutoff = datetime.now() - timedelta(days=7)
            
            tasks_to_remove = [
                task_id for task_id, task in GlobalState.tasks.items()
                if task.created_at < cutoff
            ]
            
            for task_id in tasks_to_remove:
                del GlobalState.tasks[task_id]
            
            logger.info(f"Removed {len(tasks_to_remove)} old tasks")
        
        cleanup_task = PythonOperator(
            task_id='cleanup_tasks',
            python_callable=cleanup_old_tasks
        )
    
    return [health_dag, rag_dag, cleanup_dag]


# ==============================================
# MAIN EXECUTION
# ==============================================
def main():
    """Main execution function"""
    import uvicorn
    
    # Create Airflow DAGs (in production, these would be in separate files)
    dags = create_airflow_dags()
    logger.info(f"Created {len(dags)} Airflow DAGs")
    
    # Run server
    logger.info(f"Starting FastAPI server on port {SystemConfig.NGROK_PORT}...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SystemConfig.NGROK_PORT,
        log_level="info" if not SystemConfig.DEBUG else "debug"
    )


if __name__ == "__main__":
    main()
