"""
Enterprise AI Agent Orchestration System
Architecture: Modular, Scalable, Production-Ready
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal
from urllib.parse import urljoin, urlparse

import chromadb
import httpx
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================
# CONFIGURATION & SETTINGS
# ============================================

load_dotenv()


class Config:
    """Centralized configuration management"""
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    MODEL_SERVER: str = os.getenv("MODEL_SERVER", "http://localhost:8000/v1")
    API_KEY: str = os.getenv("API_KEY", "sk-local")
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "2048"))
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    
    # RAG Configuration
    WEB_SCRAPE_URL: str = os.getenv("WEB_SCRAPE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    
    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "15"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "300"))
    
    # Paths
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "./cache"))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", "./logs"))
    
    @classmethod
    def setup_directories(cls):
        """Create required directories"""
        for dir_path in [cls.CACHE_DIR, cls.DATA_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Setup logging
logger.add(
    Config.LOGS_DIR / "agent_system_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)


# ============================================
# DATA MODELS
# ============================================

class AgentRole(str, Enum):
    """Agent role enumeration"""
    CODE_ARCHITECT = "code_architect"
    SEC_ANALYST = "sec_analyst"
    AUTO_BOT = "auto_bot"
    CREATIVE_AGENT = "creative_agent"
    AGENT_SUITE = "agent_suite"
    CUSTOM = "custom"
    ROUTER = "router"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(TypedDict):
    """LangGraph agent state"""
    messages: List[Dict[str, Any]]
    task: str
    context: Dict[str, Any]
    agent_role: str
    iteration: int
    max_iterations: int
    final_output: Optional[str]
    error: Optional[str]
    rag_context: Optional[List[str]]
    metadata: Dict[str, Any]


@dataclass
class AgentTask:
    """Agent task definition"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None


class AgentResponse(BaseModel):
    """Standardized agent response"""
    agent_role: AgentRole
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# LLM CLIENT
# ============================================

class LLMClient:
    """vLLM client with OpenAI-compatible API"""
    
    def __init__(self):
        self.base_url = Config.MODEL_SERVER
        self.api_key = Config.API_KEY
        self.model = Config.MODEL_NAME
        self.client = httpx.AsyncClient(timeout=120.0)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None
    ) -> str:
        """Generate text using vLLM server"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or Config.MODEL_MAX_TOKENS,
            "temperature": temperature or Config.MODEL_TEMPERATURE,
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# ============================================
# RAG SYSTEM - WEB SCRAPING
# ============================================

class WebScraper:
    """Advanced web scraper for RAG data collection"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_google_cse(self, query: str, max_results: int = 10) -> List[str]:
        """Scrape Google Custom Search Engine"""
        base_url = Config.WEB_SCRAPE_URL
        
        try:
            # Add query parameter
            search_url = f"{base_url}&q={query}"
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract search result links
            results = []
            for link in soup.find_all('a', href=True)[:max_results]:
                href = link['href']
                # Filter valid URLs
                if href.startswith('http') and 'google.com' not in href:
                    results.append(href)
            
            logger.info(f"Scraped {len(results)} URLs for query: {query}")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return []
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract text content from URL"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = '\n'.join(lines)
            
            logger.info(f"Extracted {len(content)} chars from {url}")
            return content
            
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return None


class RAGSystem:
    """Retrieval-Augmented Generation system with live web scraping"""
    
    def __init__(self):
        self.scraper = WebScraper()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            cache_folder=str(Config.CACHE_DIR)
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            self.vector_store = Chroma(
                persist_directory=Config.VECTOR_DB_PATH,
                embedding_function=self.embeddings,
                collection_name="agent_knowledge"
            )
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
    
    def ingest_from_web(self, query: str) -> int:
        """Scrape web and ingest into vector store"""
        urls = self.scraper.scrape_google_cse(query, Config.MAX_SEARCH_RESULTS)
        
        documents = []
        for url in urls:
            content = self.scraper.extract_content(url)
            if content:
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "chunk": i,
                            "query": query,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
        
        if documents and self.vector_store:
            self.vector_store.add_documents(documents)
            logger.info(f"Ingested {len(documents)} document chunks")
            return len(documents)
        
        return 0
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context for query"""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            contexts = [doc.page_content for doc in docs]
            logger.info(f"Retrieved {len(contexts)} context chunks")
            return contexts
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []


# ============================================
# AGENT IMPLEMENTATIONS
# ============================================

class BaseAgent(ABC):
    """Abstract base agent class"""
    
    def __init__(self, role: AgentRole, llm_client: LLMClient, rag_system: RAGSystem):
        self.role = role
        self.llm = llm_client
        self.rag = rag_system
        logger.info(f"Initialized agent: {role.value}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return agent-specific system prompt"""
        pass
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute agent task"""
        pass
    
    async def _generate_with_rag(self, task: str, rag_query: Optional[str] = None) -> str:
        """Generate response with RAG context"""
        # Retrieve context
        contexts = self.rag.retrieve(rag_query or task)
        
        # Build prompt with context
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Based on the following context, answer the task.

{context_str}

Task: {task}

Provide a comprehensive, actionable response."""
        
        return await self.llm.generate(prompt, system_prompt=self.get_system_prompt())


class CodeArchitectAgent(BaseAgent):
    """Code architecture and development agent"""
    
    def get_system_prompt(self) -> str:
        return """You are an elite Software Architect and Engineer specializing in:
- Clean, modular, production-grade code design
- Python, JavaScript, Rust, Go, API development
- Microservices, distributed systems, scalability
- Best practices: SOLID, DRY, design patterns
- Performance optimization and algorithm design

Provide code that is: well-documented, type-hinted, error-handled, testable, and enterprise-ready."""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        try:
            # Check if RAG needed
            if context.get("use_rag", False):
                response = await self._generate_with_rag(task)
            else:
                response = await self.llm.generate(task, system_prompt=self.get_system_prompt())
            
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                output=response,
                metadata={"agent": "code_architect", "context": context}
            )
        except Exception as e:
            logger.error(f"CodeArchitect execution error: {e}")
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.FAILED,
                error=str(e)
            )


class SecAnalystAgent(BaseAgent):
    """Security analysis and penetration testing agent"""
    
    def get_system_prompt(self) -> str:
        return """You are a Senior Security Analyst and Ethical Hacker specializing in:
- Security audits, penetration testing, threat modeling
- OWASP Top 10, CVE analysis, vulnerability assessment
- Network security, encryption, authentication/authorization
- Security best practices and compliance (SOC2, ISO27001)

Provide comprehensive security analysis with actionable remediation steps."""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        try:
            response = await self.llm.generate(task, system_prompt=self.get_system_prompt())
            
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                output=response,
                metadata={"agent": "sec_analyst", "security_level": "high"}
            )
        except Exception as e:
            logger.error(f"SecAnalyst execution error: {e}")
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.FAILED,
                error=str(e)
            )


class AutoBotAgent(BaseAgent):
    """Automation and workflow orchestration agent"""
    
    def get_system_prompt(self) -> str:
        return """You are an Automation Specialist focusing on:
- Workflow automation (Airflow, n8n, Zapier, Make)
- API integrations, webhooks, event-driven architecture
- CI/CD pipelines, deployment automation
- Data pipelines (Airbyte, Kafka, ETL)
- Process optimization and efficiency

Design robust, scalable automation solutions."""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        try:
            response = await self.llm.generate(task, system_prompt=self.get_system_prompt())
            
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                output=response,
                metadata={"agent": "auto_bot", "automation_type": context.get("type", "general")}
            )
        except Exception as e:
            logger.error(f"AutoBot execution error: {e}")
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.FAILED,
                error=str(e)
            )


class CreativeAgent(BaseAgent):
    """Creative content generation agent"""
    
    def get_system_prompt(self) -> str:
        return """You are a Creative Content Specialist expert in:
- Content creation: articles, blogs, marketing copy
- Visual design concepts and specifications
- Audio/video script writing
- Brand storytelling and messaging
- SEO optimization and engagement strategies

Create compelling, engaging, high-quality content."""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        try:
            response = await self.llm.generate(
                task,
                system_prompt=self.get_system_prompt(),
                temperature=0.9  # Higher creativity
            )
            
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                output=response,
                metadata={"agent": "creative", "content_type": context.get("content_type", "text")}
            )
        except Exception as e:
            logger.error(f"CreativeAgent execution error: {e}")
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.FAILED,
                error=str(e)
            )


class AgentSuiteAgent(BaseAgent):
    """Business operations and administrative agent"""
    
    def get_system_prompt(self) -> str:
        return """You are a Business Operations Manager specializing in:
- Report generation and data analysis
- Financial modeling and budgeting
- Operations automation and optimization
- Project management and planning
- Resource allocation and efficiency

Provide data-driven, actionable business insights."""
    
    async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        try:
            response = await self.llm.generate(task, system_prompt=self.get_system_prompt())
            
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                output=response,
                metadata={"agent": "agent_suite", "ops_type": context.get("ops_type", "general")}
            )
        except Exception as e:
            logger.error(f"AgentSuite execution error: {e}")
            return AgentResponse(
                agent_role=self.role,
                task_id=context.get("task_id", ""),
                status=TaskStatus.FAILED,
                error=str(e)
            )


class CustomAgentBuilder:
    """Dynamic custom agent builder"""
    
    def __init__(self, llm_client: LLMClient, rag_system: RAGSystem):
        self.llm = llm_client
        self.rag = rag_system
        self.custom_agents: Dict[str, BaseAgent] = {}
    
    def create_custom_agent(
        self,
        agent_id: str,
        system_prompt: str,
        capabilities: List[str]
    ) -> BaseAgent:
        """Create a custom agent with specific capabilities"""
        
        class DynamicCustomAgent(BaseAgent):
            def __init__(self, llm, rag, custom_prompt):
                super().__init__(AgentRole.CUSTOM, llm, rag)
                self.custom_prompt = custom_prompt
            
            def get_system_prompt(self) -> str:
                return self.custom_prompt
            
            async def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
                try:
                    response = await self.llm.generate(task, system_prompt=self.get_system_prompt())
                    return AgentResponse(
                        agent_role=self.role,
                        task_id=context.get("task_id", ""),
                        status=TaskStatus.COMPLETED,
                        output=response,
                        metadata={"agent": "custom", "agent_id": context.get("agent_id", "")}
                    )
                except Exception as e:
                    logger.error(f"CustomAgent execution error: {e}")
                    return AgentResponse(
                        agent_role=self.role,
                        task_id=context.get("task_id", ""),
                        status=TaskStatus.FAILED,
                        error=str(e)
                    )
        
        agent = DynamicCustomAgent(self.llm, self.rag, system_prompt)
        self.custom_agents[agent_id] = agent
        logger.info(f"Created custom agent: {agent_id}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve custom agent by ID"""
        return self.custom_agents.get(agent_id)


# ============================================
# AGENT ORCHESTRATION - LANGGRAPH
# ============================================

class AgentOrchestrator:
    """LangGraph-based multi-agent orchestrator"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.rag = RAGSystem()
        
        # Initialize agents
        self.agents = {
            AgentRole.CODE_ARCHITECT: CodeArchitectAgent(AgentRole.CODE_ARCHITECT, self.llm, self.rag),
            AgentRole.SEC_ANALYST: SecAnalystAgent(AgentRole.SEC_ANALYST, self.llm, self.rag),
            AgentRole.AUTO_BOT: AutoBotAgent(AgentRole.AUTO_BOT, self.llm, self.rag),
            AgentRole.CREATIVE_AGENT: CreativeAgent(AgentRole.CREATIVE_AGENT, self.llm, self.rag),
            AgentRole.AGENT_SUITE: AgentSuiteAgent(AgentRole.AGENT_SUITE, self.llm, self.rag),
        }
        
        self.custom_builder = CustomAgentBuilder(self.llm, self.rag)
        self.workflow = self._build_workflow()
        
        logger.info("AgentOrchestrator initialized with all agents")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_task)
        workflow.add_node("code_architect", self._execute_code_architect)
        workflow.add_node("sec_analyst", self._execute_sec_analyst)
        workflow.add_node("auto_bot", self._execute_auto_bot)
        workflow.add_node("creative_agent", self._execute_creative_agent)
        workflow.add_node("agent_suite", self._execute_agent_suite)
        workflow.add_node("finalize", self._finalize_result)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "code_architect": "code_architect",
                "sec_analyst": "sec_analyst",
                "auto_bot": "auto_bot",
                "creative_agent": "creative_agent",
                "agent_suite": "agent_suite",
                "end": END
            }
        )
        
        # Add edges to finalize
        for agent in ["code_architect", "sec_analyst", "auto_bot", "creative_agent", "agent_suite"]:
            workflow.add_edge(agent, "finalize")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _route_task(self, state: AgentState) -> AgentState:
        """Route task to appropriate agent"""
        task = state["task"].lower()
        
        # Simple keyword-based routing (can be enhanced with LLM-based routing)
        routing_keywords = {
            "code": AgentRole.CODE_ARCHITECT,
            "security": AgentRole.SEC_ANALYST,
            "automate": AgentRole.AUTO_BOT,
            "creative": AgentRole.CREATIVE_AGENT,
            "business": AgentRole.AGENT_SUITE,
        }
        
        for keyword, role in routing_keywords.items():
            if keyword in task:
                state["agent_role"] = role.value
                logger.info(f"Routed task to: {role.value}")
                return state
        
        # Default to code architect
        state["agent_role"] = AgentRole.CODE_ARCHITECT.value
        logger.info("Routed task to default: code_architect")
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Decide next node based on agent_role"""
        agent_role = state.get("agent_role", "")
        
        if state.get("error"):
            return "end"
        
        return agent_role if agent_role else "end"
    
    async def _execute_code_architect(self, state: AgentState) -> AgentState:
        """Execute CodeArchitect agent"""
        agent = self.agents[AgentRole.CODE_ARCHITECT]
        response = await agent.execute(state["task"], state["context"])
        state["final_output"] = response.output
        state["metadata"]["agent_response"] = response.dict()
        return state
    
    async def _execute_sec_analyst(self, state: AgentState) -> AgentState:
        """Execute SecAnalyst agent"""
        agent = self.agents[AgentRole.SEC_ANALYST]
        response = await agent.execute(state["task"], state["context"])
        state["final_output"] = response.output
        state["metadata"]["agent_response"] = response.dict()
        return state
    
    async def _execute_auto_bot(self, state: AgentState) -> AgentState:
        """Execute AutoBot agent"""
        agent = self.agents[AgentRole.AUTO_BOT]
        response = await agent.execute(state["task"], state["context"])
        state["final_output"] = response.output
        state["metadata"]["agent_response"] = response.dict()
        return state
    
    async def _execute_creative_agent(self, state: AgentState) -> AgentState:
        """Execute Creative agent"""
        agent = self.agents[AgentRole.CREATIVE_AGENT]
        response = await agent.execute(state["task"], state["context"])
        state["final_output"] = response.output
        state["metadata"]["agent_response"] = response.dict()
        return state
    
    async def _execute_agent_suite(self, state: AgentState) -> AgentState:
        """Execute AgentSuite agent"""
        agent = self.agents[AgentRole.AGENT_SUITE]
        response = await agent.execute(state["task"], state["context"])
        state["final_output"] = response.output
        state["metadata"]["agent_response"] = response.dict()
        return state
    
    async def _finalize_result(self, state: AgentState) -> AgentState:
        """Finalize and format result"""
        state["iteration"] += 1
        state["metadata"]["completed_at"] = datetime.now().isoformat()
        logger.info(f"Task completed by {state['agent_role']}")
        return state
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        use_rag: bool = False,
        rag_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute task through agent workflow"""
        
        # Ingest RAG data if needed
        if use_rag and rag_query:
            self.rag.ingest_from_web(rag_query)
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [],
            "task": task,
            "context": context or {},
            "agent_role": "",
            "iteration": 0,
            "max_iterations": Config.MAX_AGENT_ITERATIONS,
            "final_output": None,
            "error": None,
            "rag_context": None,
            "metadata": {
                "started_at": datetime.now().isoformat(),
                "use_rag": use_rag
            }
        }
        
        try:
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.llm.close()


# ============================================
# MAIN SYSTEM CLASS
# ============================================

class AIAgentPlatform:
    """Main AI Agent Platform orchestrator"""
    
    def __init__(self):
        Config.setup_directories()
        self.orchestrator = AgentOrchestrator()
        self.tasks: Dict[str, AgentTask] = {}
        logger.info("AI Agent Platform initialized")
    
    async def create_task(
        self,
        task_type: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AgentTask:
        """Create new agent task"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
        
        task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            parameters=parameters or {}
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task: {task_id}")
        return task
    
    async def execute_task(self, task_id: str) -> AgentTask:
        """Execute task by ID"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            result = await self.orchestrator.execute_task(
                task=task.description,
                context={"task_id": task_id, **task.parameters},
                use_rag=task.parameters.get("use_rag", False),
                rag_query=task.parameters.get("rag_query")
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task execution failed: {task_id} - {e}")
        
        return task
    
    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[AgentTask]:
        """List all tasks"""
        return list(self.tasks.values())
    
    async def shutdown(self):
        """Shutdown platform"""
        await self.orchestrator.cleanup()
        logger.info("Platform shutdown complete")


# ============================================
# SINGLETON INSTANCE
# ============================================

_platform_instance: Optional[AIAgentPlatform] = None

def get_platform() -> AIAgentPlatform:
    """Get or create platform singleton"""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = AIAgentPlatform()
    return _platform_instance
