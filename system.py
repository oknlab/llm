"""
ELITE MULTI-AGENT ORCHESTRATION SYSTEM
Architecture: Clean, Modular, Enterprise-Grade
Pattern: Agentic RAG + LangGraph State Machine
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import operator
from datetime import datetime
import json
import logging

# Core AI Stack
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Web Scraping
import requests
from bs4 import BeautifulSoup
import html2text

# Model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Utils
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================
# LOGGING CONFIGURATION
# ==============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================
# ENUMS & CONSTANTS
# ==============================================
class AgentType(Enum):
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"
    AUTO_BOT = "AutoBot"
    AGENT_SUITE = "AgentSuite"
    CREATIVE_AGENT = "CreativeAgent"
    AG_CUSTOM = "AgCustom"
    ORCHESTRATOR = "Orchestrator"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================
# DATA MODELS
# ==============================================
@dataclass
class AgentConfig:
    """Agent configuration with enterprise defaults"""
    name: str
    type: AgentType
    system_prompt: str
    max_iterations: int = 10
    temperature: float = 0.7
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True


@dataclass
class Task:
    """Task execution model"""
    id: str
    agent_type: AgentType
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class AgentState(TypedDict):
    """LangGraph state schema"""
    messages: Annotated[Sequence[str], operator.add]
    current_agent: str
    task: Task
    rag_context: List[Document]
    iterations: int
    final_output: Optional[str]
    metadata: Dict[str, Any]


# ==============================================
# RAG PIPELINE - WEB SCRAPING ENGINE
# ==============================================
class WebScrapingRAG:
    """Production-grade RAG with real web scraping"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(config.get('CHUNK_SIZE', 1000)),
            chunk_overlap=int(config.get('CHUNK_OVERLAP', 200))
        )
        self.vector_store = None
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embedding model"""
        model_name = self.config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def scrape_url(self, url: str) -> Optional[Document]:
        """Scrape single URL with retry logic"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract text
            text = self.html_converter.handle(str(soup))
            
            return Document(
                page_content=text,
                metadata={"source": url, "scraped_at": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def google_cse_search(self, query: str, num_results: int = 5) -> List[str]:
        """Search using Google Custom Search Engine"""
        try:
            cse_id = self.config.get('GOOGLE_CSE_ID')
            api_key = self.config.get('GOOGLE_API_KEY')
            
            if not cse_id or not api_key:
                logger.warning("Google CSE not configured, using fallback search")
                return self._fallback_search(query, num_results)
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': cse_id,
                'q': query,
                'num': num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            urls = [item['link'] for item in data.get('items', [])]
            return urls
        except Exception as e:
            logger.error(f"Google CSE error: {e}")
            return self._fallback_search(query, num_results)
    
    def _fallback_search(self, query: str, num_results: int) -> List[str]:
        """Fallback search mechanism"""
        # Using DuckDuckGo HTML scraping as fallback
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            for link in soup.find_all('a', class_='result__url', limit=num_results):
                href = link.get('href')
                if href and href.startswith('http'):
                    results.append(href)
            return results
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    def build_rag_index(self, query: str) -> bool:
        """Build RAG index from web scraping"""
        logger.info(f"Building RAG index for query: {query}")
        
        # Get URLs from search
        urls = self.google_cse_search(query, num_results=int(self.config.get('TOP_K_RESULTS', 5)))
        
        if not urls:
            logger.warning("No URLs found for query")
            return False
        
        # Parallel scraping
        documents = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.scrape_url, url): url for url in urls}
            for future in as_completed(future_to_url):
                doc = future.result()
                if doc:
                    documents.append(doc)
        
        if not documents:
            logger.warning("No documents scraped successfully")
            return False
        
        # Split and embed
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        return True
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        return self.vector_store.similarity_search(query, k=k)


# ==============================================
# MODEL MANAGER
# ==============================================
class ModelManager:
    """Singleton model manager for Qwen"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _pipeline = None
    
    def __new__(cls, config: Dict[str, Any]):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config)
        return cls._instance
    
    def _initialize(self, config: Dict[str, Any]):
        """Initialize model once"""
        logger.info("Initializing Qwen model...")
        
        model_name = config.get('MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct')
        device = config.get('MODEL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with quantization
        load_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32
        }
        
        if config.get('MODEL_QUANTIZATION') == '8bit' and device == 'cuda':
            load_kwargs['load_in_8bit'] = True
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Create pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=int(config.get('MAX_NEW_TOKENS', 2048)),
            temperature=float(config.get('TEMPERATURE', 0.7)),
            top_p=float(config.get('TOP_P', 0.95)),
            do_sample=True,
            device=0 if device == 'cuda' else -1
        )
        
        logger.info(f"Model loaded on {device}")
    
    def get_pipeline(self) -> Any:
        """Get HuggingFace pipeline"""
        return self._pipeline
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text"""
        try:
            result = self._pipeline(prompt, **kwargs)
            return result[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"


# ==============================================
# AGENT DEFINITIONS
# ==============================================
class BaseAgent:
    """Base agent with common functionality"""
    
    def __init__(self, config: AgentConfig, model_manager: ModelManager, rag: WebScrapingRAG):
        self.config = config
        self.model = model_manager
        self.rag = rag
        self.memory: List[Dict[str, str]] = []
    
    def _build_prompt(self, query: str, context: List[Document]) -> str:
        """Build prompt with RAG context"""
        context_text = "\n\n".join([doc.page_content[:500] for doc in context[:3]])
        
        prompt = f"""<|im_start|>system
{self.config.system_prompt}
<|im_end|>

<|im_start|>context
{context_text if context else 'No additional context available.'}
<|im_end|>

<|im_start|>user
{query}
<|im_end|>

<|im_start|>assistant
"""
        return prompt
    
    def execute(self, task: Task, rag_context: List[Document]) -> str:
        """Execute agent task"""
        logger.info(f"[{self.config.name}] Executing task: {task.query[:100]}")
        
        prompt = self._build_prompt(task.query, rag_context)
        
        response = self.model.generate(
            prompt,
            max_new_tokens=int(os.getenv('MAX_NEW_TOKENS', 2048)),
            temperature=self.config.temperature
        )
        
        # Store in memory
        if self.config.memory_enabled:
            self.memory.append({
                "query": task.query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            # Keep last N messages
            self.memory = self.memory[-int(os.getenv('MEMORY_WINDOW', 10)):]
        
        return response


# ==============================================
# SPECIALIZED AGENTS
# ==============================================
AGENT_PROMPTS = {
    AgentType.CODE_ARCHITECT: """You are CodeArchitect, an elite software engineer specializing in:
- Complex system design and architecture
- Multi-language development (Python, JavaScript, Rust, Go, Java, C++)
- API design and microservices
- Performance optimization and scalability
- Clean code principles and design patterns

Provide production-ready, optimized code with proper error handling and documentation.""",

    AgentType.SEC_ANALYST: """You are SecAnalyst, a cybersecurity expert specializing in:
- Penetration testing and vulnerability assessment
- Security audits and code review
- Threat modeling and risk analysis
- Compliance and security best practices
- Incident response and forensics

Provide actionable security recommendations with severity ratings.""",

    AgentType.AUTO_BOT: """You are AutoBot, an automation specialist focusing on:
- API integration and orchestration
- Workflow automation (n8n, Zapier, Make)
- CI/CD pipeline optimization
- Infrastructure as Code
- Process automation and optimization

Design scalable, maintainable automation solutions.""",

    AgentType.AGENT_SUITE: """You are AgentSuite, an administrative agent handling:
- Schedule management and coordination
- Report generation and analysis
- Financial data processing
- Operational workflow optimization
- Resource allocation

Provide structured, actionable administrative solutions.""",

    AgentType.CREATIVE_AGENT: """You are CreativeAgent, a creative specialist for:
- High-quality content generation (text, scripts, copy)
- Visual concept development
- Audio content planning
- Brand and marketing materials
- Creative strategy

Generate original, engaging content aligned with objectives.""",

    AgentType.AG_CUSTOM: """You are AgCustom, a custom agent builder specializing in:
- Custom AI agent development
- Domain-specific solutions
- Integration with existing systems
- Specialized workflow creation
- Adaptive problem-solving

Design tailored solutions for unique requirements."""
}


def create_agent(agent_type: AgentType, model_manager: ModelManager, rag: WebScrapingRAG) -> BaseAgent:
    """Factory function for agent creation"""
    config = AgentConfig(
        name=agent_type.value,
        type=agent_type,
        system_prompt=AGENT_PROMPTS[agent_type],
        temperature=0.7 if agent_type != AgentType.CREATIVE_AGENT else 0.9
    )
    return BaseAgent(config, model_manager, rag)


# ==============================================
# LANGGRAPH ORCHESTRATION
# ==============================================
class AgentOrchestrator:
    """LangGraph-based multi-agent orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag = WebScrapingRAG(config)
        self.model_manager = ModelManager(config)
        
        # Initialize all agents
        self.agents = {
            agent_type: create_agent(agent_type, self.model_manager, self.rag)
            for agent_type in AgentType
            if agent_type != AgentType.ORCHESTRATOR
        }
        
        # Build LangGraph
        self.graph = self._build_graph()
        
        logger.info("Agent Orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        for agent_type, agent in self.agents.items():
            workflow.add_node(
                agent_type.value,
                lambda state, a=agent: self._agent_node(state, a)
            )
        
        # Add routing node
        workflow.add_node("route", self._route_node)
        
        # Set entry point
        workflow.set_entry_point("route")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {agent_type.value: agent_type.value for agent_type in self.agents.keys()}
        )
        
        # All agents return to router or end
        for agent_type in self.agents.keys():
            workflow.add_conditional_edges(
                agent_type.value,
                lambda state: "route" if state['iterations'] < int(os.getenv('MAX_ITERATIONS', 10)) else END,
                {"route": "route", END: END}
            )
        
        return workflow.compile()
    
    def _route_node(self, state: AgentState) -> AgentState:
        """Router node - determines which agent to use"""
        task = state['task']
        state['current_agent'] = task.agent_type.value
        state['iterations'] = state.get('iterations', 0) + 1
        return state
    
    def _agent_node(self, state: AgentState, agent: BaseAgent) -> AgentState:
        """Agent execution node"""
        task = state['task']
        rag_context = state.get('rag_context', [])
        
        # Execute agent
        result = agent.execute(task, rag_context)
        
        # Update state
        state['messages'].append(f"[{agent.config.name}]: {result}")
        state['final_output'] = result
        state['metadata']['last_agent'] = agent.config.name
        state['metadata']['completed_at'] = datetime.now().isoformat()
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine which agent to route to"""
        return state['current_agent']
    
    def process_task(self, task: Task, use_rag: bool = True) -> Dict[str, Any]:
        """Process task through agent system"""
        logger.info(f"Processing task {task.id} with agent {task.agent_type.value}")
        
        # Build RAG index if enabled
        rag_context = []
        if use_rag:
            self.rag.build_rag_index(task.query)
            rag_context = self.rag.retrieve(task.query)
        
        # Initialize state
        initial_state: AgentState = {
            'messages': [],
            'current_agent': task.agent_type.value,
            'task': task,
            'rag_context': rag_context,
            'iterations': 0,
            'final_output': None,
            'metadata': {
                'task_id': task.id,
                'started_at': datetime.now().isoformat(),
                'rag_docs': len(rag_context)
            }
        }
        
        # Execute graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                'status': 'success',
                'result': final_state['final_output'],
                'metadata': final_state['metadata'],
                'iterations': final_state['iterations']
            }
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'metadata': initial_state['metadata']
            }


# ==============================================
# EXPORT
# ==============================================
__all__ = [
    'AgentOrchestrator',
    'AgentType',
    'Task',
    'TaskStatus',
    'WebScrapingRAG',
    'ModelManager'
]
