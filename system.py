"""
Enterprise AI Agent Orchestration System
Modular, Scalable, Production-Grade Multi-Agent Framework
"""

import os
import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Core Imports
from pydantic import BaseModel, Field
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain & LangGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Vector Store & RAG
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Web Scraping
import requests
from bs4 import BeautifulSoup
import httpx

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import psutil

# Configuration
from dotenv import load_dotenv
load_dotenv()


# ==================== METRICS ====================
AGENT_REQUESTS = Counter('agent_requests_total', 'Total agent requests', ['agent_type'])
AGENT_LATENCY = Histogram('agent_latency_seconds', 'Agent processing time', ['agent_type'])
AGENT_ERRORS = Counter('agent_errors_total', 'Agent errors', ['agent_type', 'error_type'])
ACTIVE_AGENTS = Gauge('active_agents', 'Number of active agents')
RAG_QUERIES = Counter('rag_queries_total', 'Total RAG queries')
CACHE_HITS = Counter('cache_hits_total', 'Cache hit count')


# ==================== ENUMS & MODELS ====================
class AgentType(str, Enum):
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"
    AUTO_BOT = "AutoBot"
    AGENT_SUITE = "AgentSuite"
    CREATIVE_AGENT = "CreativeAgent"
    AG_CUSTOM = "AgCustom"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentState(TypedDict):
    """LangGraph State Schema"""
    task: str
    agent_type: str
    context: Dict[str, Any]
    rag_results: List[Dict]
    agent_outputs: Dict[str, Any]
    routing_decision: str
    metadata: Dict[str, Any]
    iteration: int
    max_iterations: int


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest())
    description: str
    agent_type: Optional[AgentType] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


# ==================== WEB SCRAPER (RAG DATA SOURCE) ====================
class LiveWebScraper:
    """Production web scraper with rate limiting and caching"""
    
    def __init__(self):
        self.cse_url = os.getenv("GOOGLE_CSE_URL")
        self.api_key = os.getenv("GOOGLE_CSE_API_KEY")
        self.cx = os.getenv("GOOGLE_CSE_CX")
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search_google_cse(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search using Google Custom Search Engine API"""
        RAG_QUERIES.inc()
        
        cache_key = hashlib.md5(f"{query}_{num_results}".encode()).hexdigest()
        if cache_key in self.cache:
            CACHE_HITS.inc()
            logger.info(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {
                    "key": self.api_key,
                    "cx": self.cx,
                    "q": query,
                    "num": num_results
                }
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": "google_cse"
                    })
                
                self.cache[cache_key] = results
                logger.info(f"Scraped {len(results)} results for: {query}")
                return results
                
        except Exception as e:
            logger.error(f"Scraping error: {str(e)}")
            AGENT_ERRORS.inc(agent_type="rag", error_type="scraping")
            return []
    
    async def fetch_page_content(self, url: str) -> str:
        """Fetch and parse page content"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text(separator='\n', strip=True)
                return text[:5000]  # Limit content size
                
        except Exception as e:
            logger.error(f"Content fetch error for {url}: {str(e)}")
            return ""


# ==================== RAG SYSTEM ====================
class RAGPipeline:
    """Production RAG with FAISS and live web scraping"""
    
    def __init__(self):
        self.scraper = LiveWebScraper()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL"),
            model_kwargs={'device': os.getenv("DEVICE", "cpu")}
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
        
    async def build_knowledge_base(self, query: str) -> None:
        """Build vector store from live scraped data"""
        logger.info(f"Building knowledge base for: {query}")
        
        # Scrape live data
        search_results = await self.scraper.search_google_cse(query)
        
        if not search_results:
            logger.warning("No search results found")
            return
        
        # Fetch full content
        documents = []
        for result in search_results:
            content = await self.scraper.fetch_page_content(result["link"])
            if content:
                documents.append({
                    "content": content,
                    "metadata": {
                        "title": result["title"],
                        "url": result["link"],
                        "snippet": result["snippet"]
                    }
                })
        
        if not documents:
            logger.warning("No documents fetched")
            return
        
        # Split and embed
        texts = []
        metadatas = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            texts.extend(chunks)
            metadatas.extend([doc["metadata"]] * len(chunks))
        
        # Create FAISS index
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        logger.info(f"Built vector store with {len(texts)} chunks")
    
    async def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        if not self.vector_store:
            await self.build_knowledge_base(query)
        
        if not self.vector_store:
            return []
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0
            }
            for doc in docs
        ]


# ==================== LLM ENGINE ====================
class QwenEngine:
    """Optimized Qwen LLM with caching and batching"""
    
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.device = os.getenv("DEVICE", "cpu")
        self.max_length = int(os.getenv("MAX_LENGTH", 4096))
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Lazy load model with quantization"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        
        if self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            do_sample=True,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("Model loaded successfully")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate text with context"""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        try:
            result = self.pipeline(
                full_prompt,
                max_new_tokens=512,
                return_full_text=False
            )
            return result[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            AGENT_ERRORS.inc(agent_type="llm", error_type="generation")
            raise


# ==================== SPECIALIZED AGENTS ====================
class BaseAgent:
    """Base agent with common functionality"""
    
    def __init__(self, agent_type: AgentType, llm_engine: QwenEngine, rag: RAGPipeline):
        self.agent_type = agent_type
        self.llm = llm_engine
        self.rag = rag
        self.system_prompt = self._get_system_prompt()
        
    def _get_system_prompt(self) -> str:
        """Agent-specific system prompts"""
        prompts = {
            AgentType.CODE_ARCHITECT: """You are CodeArchitect, an elite software engineer specializing in:
- Clean, modular, production-grade code (Python, JS, Rust, Go)
- System architecture and API design
- Performance optimization and scalability
- Code audits and refactoring
Respond with precise, actionable code solutions.""",

            AgentType.SEC_ANALYST: """You are SecAnalyst, a cybersecurity expert specializing in:
- Penetration testing and vulnerability assessment
- Security audits and threat modeling
- Secure coding practices
- Compliance (OWASP, NIST, ISO 27001)
Provide detailed security analysis and remediation steps.""",

            AgentType.AUTO_BOT: """You are AutoBot, an automation specialist focused on:
- API integrations (REST, GraphQL, webhooks)
- Workflow automation (Zapier, n8n, Make)
- Data pipeline orchestration (Airflow, Kafka)
- Process optimization
Design scalable, reliable automation solutions.""",

            AgentType.AGENT_SUITE: """You are AgentSuite, handling admin operations:
- Report generation and analytics
- Financial automation and tracking
- Operational workflows
- Documentation and knowledge management
Deliver structured, actionable administrative outputs.""",

            AgentType.CREATIVE_AGENT: """You are CreativeAgent, a content specialist for:
- Technical writing and documentation
- Visual content descriptions
- Narrative design for user experiences
- Multi-modal content strategy
Create engaging, clear, professional content.""",

            AgentType.AG_CUSTOM: """You are AgCustom, a versatile AI agent that:
- Adapts to custom requirements
- Combines capabilities from other agents
- Handles edge cases and unique workflows
Provide flexible, context-aware solutions."""
        }
        return prompts.get(self.agent_type, "You are a helpful AI assistant.")
    
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task with RAG"""
        AGENT_REQUESTS.inc(agent_type=self.agent_type.value)
        
        with AGENT_LATENCY.labels(agent_type=self.agent_type.value).time():
            try:
                # Retrieve relevant context
                rag_results = await self.rag.retrieve(task)
                rag_context = "\n\n".join([
                    f"[{r['metadata'].get('title', 'Source')}]\n{r['content'][:500]}"
                    for r in rag_results[:3]
                ])
                
                # Build prompt
                prompt = f"""System: {self.system_prompt}

Context from knowledge base:
{rag_context}

Task: {task}

Additional Context: {json.dumps(context, indent=2)}

Provide a detailed, actionable response:"""
                
                # Generate response
                response = await self.llm.generate(prompt)
                
                return {
                    "agent": self.agent_type.value,
                    "response": response,
                    "rag_sources": [r["metadata"] for r in rag_results],
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Agent {self.agent_type.value} error: {str(e)}")
                AGENT_ERRORS.inc(agent_type=self.agent_type.value, error_type="execution")
                return {
                    "agent": self.agent_type.value,
                    "error": str(e),
                    "success": False
                }


# ==================== ROUTER AGENT ====================
class RouterAgent:
    """Intelligent task router using LLM"""
    
    def __init__(self, llm_engine: QwenEngine):
        self.llm = llm_engine
        
    async def route(self, task: str) -> AgentType:
        """Route task to appropriate agent"""
        prompt = f"""Analyze this task and determine the most appropriate agent:

Task: {task}

Agents:
- CodeArchitect: Software development, coding, architecture
- SecAnalyst: Security, audits, vulnerabilities
- AutoBot: Automation, integrations, workflows
- AgentSuite: Admin, reports, operations
- CreativeAgent: Content creation, writing
- AgCustom: Custom/hybrid requirements

Respond with ONLY the agent name (e.g., CodeArchitect):"""
        
        try:
            response = await self.llm.generate(prompt)
            agent_name = response.strip().split()[0]
            
            # Map to enum
            for agent_type in AgentType:
                if agent_type.value.lower() == agent_name.lower():
                    logger.info(f"Routed to: {agent_type.value}")
                    return agent_type
            
            # Default fallback
            return AgentType.AG_CUSTOM
            
        except Exception as e:
            logger.error(f"Routing error: {str(e)}")
            return AgentType.AG_CUSTOM


# ==================== LANGGRAPH ORCHESTRATOR ====================
class AgentOrchestrator:
    """LangGraph-based multi-agent orchestration"""
    
    def __init__(self):
        self.llm_engine = QwenEngine()
        self.rag = RAGPipeline()
        self.router = RouterAgent(self.llm_engine)
        
        # Initialize agents
        self.agents = {
            agent_type: BaseAgent(agent_type, self.llm_engine, self.rag)
            for agent_type in AgentType
        }
        
        # Build graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("route", self._route_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Edges
        workflow.set_entry_point("route")
        workflow.add_edge("route", "execute")
        workflow.add_edge("execute", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def _route_node(self, state: AgentState) -> AgentState:
        """Routing node"""
        agent_type = await self.router.route(state["task"])
        state["routing_decision"] = agent_type.value
        state["metadata"]["routed_at"] = datetime.now().isoformat()
        return state
    
    async def _execute_node(self, state: AgentState) -> AgentState:
        """Execution node"""
        agent_type = AgentType(state["routing_decision"])
        agent = self.agents[agent_type]
        
        result = await agent.execute(state["task"], state["context"])
        state["agent_outputs"][agent_type.value] = result
        state["rag_results"] = result.get("rag_sources", [])
        
        return state
    
    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesis node"""
        state["metadata"]["completed_at"] = datetime.now().isoformat()
        state["metadata"]["total_agents"] = len(state["agent_outputs"])
        return state
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process task through graph"""
        ACTIVE_AGENTS.inc()
        
        try:
            initial_state = AgentState(
                task=task.description,
                agent_type="",
                context=task.context,
                rag_results=[],
                agent_outputs={},
                routing_decision="",
                metadata={
                    "task_id": task.task_id,
                    "priority": task.priority.value,
                    "started_at": datetime.now().isoformat()
                },
                iteration=0,
                max_iterations=1
            )
            
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "task_id": task.task_id,
                "status": "completed",
                "result": result["agent_outputs"],
                "routing": result["routing_decision"],
                "rag_sources": result["rag_results"],
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Task processing error: {str(e)}")
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e)
            }
        finally:
            ACTIVE_AGENTS.dec()


# ==================== SYSTEM MANAGER ====================
class SystemManager:
    """Main system coordinator"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", 4)))
        logger.info("System Manager initialized")
        
    async def submit_task(self, task: Task) -> Dict[str, Any]:
        """Submit task for processing"""
        logger.info(f"Processing task {task.task_id}: {task.description}")
        return await self.orchestrator.process_task(task)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down system...")
        self.executor.shutdown(wait=True)


# ==================== SINGLETON ====================
_system_instance = None

def get_system() -> SystemManager:
    """Get or create system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = SystemManager()
    return _system_instance


if __name__ == "__main__":
    # Test the system
    async def test():
        system = get_system()
        task = Task(
            description="Audit the security of a FastAPI application with JWT authentication",
            agent_type=AgentType.SEC_ANALYST,
            priority=TaskPriority.HIGH,
            context={"framework": "FastAPI", "auth": "JWT"}
        )
        result = await system.submit_task(task)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
