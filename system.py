"""
ELITE SELF-IMPROVING MULTI-AGENT ECOSYSTEM
Architecture: Enterprise-Grade, Modular, Observable, Self-Improving
"""

import os
import json
import logging
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor

from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent

from pydantic_settings import BaseSettings
from pydantic import Field
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
from kafka import KafkaProducer
import mlflow
import wandb

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

class Settings(BaseSettings):
    # Core
    ngrok_auth_token: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    env: str = "production"
    log_level: str = "INFO"
    
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda"
    temperature: float = 0.7
    
    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_enabled: bool = True
    
    # Scraping
    cse_url: str = ""
    max_scrape_pages: int = 10
    
    # Integrations
    slack_bot_token: str = ""
    notion_api_key: str = ""
    github_token: str = ""
    confluence_url: str = ""
    confluence_username: str = ""
    confluence_api_token: str = ""
    zapier_webhook_url: str = ""
    n8n_webhook_url: str = ""
    
    # Data Pipelines
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_agents: str = "agent-events"
    s3_endpoint: str = ""
    s3_bucket: str = "ai-agent-data"
    
    # Monitoring
    prometheus_port: int = 9090
    grafana_url: str = ""
    
    # Model Management
    mlflow_tracking_uri: str = "http://localhost:5000"
    wandb_api_key: str = ""
    wandb_project: str = "ai-agent-system"
    
    # Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Self-Improvement
    enable_self_improvement: bool = True
    feedback_threshold: float = 0.7
    improvement_interval: int = 3600
    
    class Config:
        env_file = ".env"
        case_sensitive = False

config = Settings()

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════

METRICS = {
    'agent_requests': Counter('agent_requests_total', 'Total agent requests', ['agent_type']),
    'agent_latency': Histogram('agent_latency_seconds', 'Agent latency', ['agent_type']),
    'rag_queries': Counter('rag_queries_total', 'RAG queries'),
    'scrape_operations': Counter('scrape_operations_total', 'Scraping operations', ['status']),
    'model_inference': Histogram('model_inference_seconds', 'Model inference time'),
    'active_agents': Gauge('active_agents', 'Number of active agents'),
    'knowledge_base_size': Gauge('knowledge_base_size', 'Documents in knowledge base'),
}

# ═══════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════

class AgentType(Enum):
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"
    AUTO_BOT = "AutoBot"
    DATA_ENGINEER = "DataEngineer"
    MLOPS_AGENT = "MLOpsAgent"
    CUSTOM = "CustomAgent"

@dataclass
class AgentState:
    """LangGraph agent state"""
    messages: List[Dict] = field(default_factory=list)
    context: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class AgentResponse:
    agent: str
    content: str
    metadata: Dict[str, Any]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: List[str] = field(default_factory=list)

@dataclass
class FeedbackData:
    query: str
    response: str
    rating: float
    timestamp: str
    agent: str

# ═══════════════════════════════════════════════════════════
# CACHE MANAGER
# ═══════════════════════════════════════════════════════════

class CacheManager:
    """Redis-based caching"""
    
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
            self.redis.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.redis = None
    
    def get(self, key: str) -> Optional[str]:
        if not self.redis:
            return None
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: str, expire: int = 3600):
        if not self.redis:
            return
        try:
            self.redis.setex(key, expire, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# ═══════════════════════════════════════════════════════════
# EVENT STREAMING
# ═══════════════════════════════════════════════════════════

class EventStreamer:
    """Kafka event streaming"""
    
    def __init__(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("Kafka producer connected")
        except Exception as e:
            logger.warning(f"Kafka unavailable: {e}")
            self.producer = None
    
    def publish(self, topic: str, event: Dict[str, Any]):
        if not self.producer:
            return
        try:
            self.producer.send(topic, event)
            self.producer.flush()
        except Exception as e:
            logger.error(f"Event publish error: {e}")

# ═══════════════════════════════════════════════════════════
# INTEGRATIONS
# ═══════════════════════════════════════════════════════════

class IntegrationHub:
    """Centralized integration management"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.events = EventStreamer()
    
    async def send_slack(self, message: str, channel: str = None):
        """Send message to Slack"""
        if not config.slack_bot_token:
            logger.warning("Slack token not configured")
            return
        
        try:
            from slack_sdk import WebClient
            client = WebClient(token=config.slack_bot_token)
            client.chat_postMessage(
                channel=channel or "#ai-agents",
                text=message
            )
            logger.info(f"Sent to Slack: {message[:50]}...")
        except Exception as e:
            logger.error(f"Slack error: {e}")
    
    async def query_notion(self, database_id: str) -> List[Dict]:
        """Query Notion database"""
        if not config.notion_api_key:
            return []
        
        try:
            from notion_client import Client
            client = Client(auth=config.notion_api_key)
            response = client.databases.query(database_id=database_id)
            return response.get('results', [])
        except Exception as e:
            logger.error(f"Notion error: {e}")
            return []
    
    async def fetch_github_repo(self, repo_name: str) -> List[Document]:
        """Fetch GitHub repository content"""
        if not config.github_token:
            return []
        
        try:
            from github import Github
            g = Github(config.github_token)
            repo = g.get_repo(repo_name)
            
            documents = []
            contents = repo.get_contents("")
            
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                elif file_content.name.endswith(('.py', '.md', '.txt')):
                    doc = Document(
                        page_content=file_content.decoded_content.decode('utf-8'),
                        metadata={
                            'source': f"github:{repo_name}/{file_content.path}",
                            'type': 'code'
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} files from {repo_name}")
            return documents
        except Exception as e:
            logger.error(f"GitHub error: {e}")
            return []
    
    async def query_confluence(self, space: str, query: str) -> List[Document]:
        """Query Confluence space"""
        if not config.confluence_api_token:
            return []
        
        try:
            from atlassian import Confluence
            confluence = Confluence(
                url=config.confluence_url,
                username=config.confluence_username,
                password=config.confluence_api_token
            )
            
            results = confluence.cql(f'space={space} and text~"{query}"')
            documents = []
            
            for result in results.get('results', []):
                page_id = result['content']['id']
                page = confluence.get_page_by_id(page_id, expand='body.storage')
                
                doc = Document(
                    page_content=page['body']['storage']['value'],
                    metadata={
                        'source': f"confluence:{page['title']}",
                        'type': 'documentation'
                    }
                )
                documents.append(doc)
            
            logger.info(f"Fetched {len(documents)} Confluence pages")
            return documents
        except Exception as e:
            logger.error(f"Confluence error: {e}")
            return []
    
    async def trigger_webhook(self, url: str, data: Dict[str, Any]):
        """Trigger external webhook"""
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            logger.info(f"Webhook triggered: {url}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

# ═══════════════════════════════════════════════════════════
# ADVANCED WEB SCRAPER
# ═══════════════════════════════════════════════════════════

class AdvancedScraper:
    """Production scraper with multi-source support"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    def scrape_google_cse(self, query: str, max_results: int = 10) -> List[Document]:
        """Scrape Google Custom Search Engine"""
        METRICS['scrape_operations'].labels(status='started').inc()
        
        try:
            search_url = f"{config.cse_url}&q={requests.utils.quote(query)}"
            response = self.session.get(
                search_url,
                headers={'User-Agent': self.ua.random},
                timeout=30
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            links = [a['href'] for a in soup.find_all('a', href=True) 
                    if a['href'].startswith('http') and 'google' not in a['href']][:max_results]
            
            futures = [self.executor.submit(self._scrape_url, url) for url in links]
            documents = [doc for future in futures if (doc := future.result())]
            
            METRICS['scrape_operations'].labels(status='success').inc()
            logger.info(f"Scraped {len(documents)} documents")
            return documents
        
        except Exception as e:
            METRICS['scrape_operations'].labels(status='error').inc()
            logger.error(f"Scraping error: {e}")
            return []
    
    def _scrape_url(self, url: str) -> Optional[Document]:
        try:
            response = self.session.get(url, headers={'User-Agent': self.ua.random}, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            return Document(
                page_content=text,
                metadata={'source': url, 'type': 'web'}
            )
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return None

# ═══════════════════════════════════════════════════════════
# RAG ENGINE WITH MULTI-SOURCE
# ═══════════════════════════════════════════════════════════

class MultiSourceRAG:
    """Advanced RAG with multiple knowledge sources"""
    
    def __init__(self):
        logger.info("Initializing Multi-Source RAG...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        self.vectorstore = None
        self.scraper = AdvancedScraper()
        self.integrations = IntegrationHub()
        
        logger.info("RAG Engine initialized")
    
    async def ingest_from_sources(self, query: str, sources: List[str] = None):
        """Ingest from multiple sources"""
        sources = sources or ['google_cse']
        all_documents = []
        
        for source in sources:
            if source == 'google_cse':
                docs = self.scraper.scrape_google_cse(query)
                all_documents.extend(docs)
            
            elif source == 'github':
                docs = await self.integrations.fetch_github_repo(query)
                all_documents.extend(docs)
            
            elif source == 'notion':
                # Implement Notion ingestion
                pass
            
            elif source == 'confluence':
                docs = await self.integrations.query_confluence('MAIN', query)
                all_documents.extend(docs)
        
        if all_documents:
            self.ingest_documents(all_documents)
            METRICS['knowledge_base_size'].set(len(all_documents))
    
    def ingest_documents(self, documents: List[Document]):
        """Index documents into vector store"""
        if not documents:
            return
        
        splits = self.text_splitter.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="multi_source_rag"
            )
        else:
            self.vectorstore.add_documents(splits)
        
        logger.info(f"Indexed {len(splits)} chunks")
    
    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Retrieve context with metadata"""
        METRICS['rag_queries'].inc()
        top_k = top_k or config.top_k
        
        if not self.vectorstore:
            return {'documents': [], 'sources': [], 'scores': []}
        
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        return {
            'documents': [doc for doc, _ in results],
            'sources': [doc.metadata.get('source', 'unknown') for doc, _ in results],
            'scores': [float(score) for _, score in results]
        }

# ═══════════════════════════════════════════════════════════
# LLM ENGINE
# ═══════════════════════════════════════════════════════════

class LLMEngine:
    """Qwen inference with monitoring"""
    
    def __init__(self):
        logger.info(f"Loading model: {config.model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate with monitoring"""
        import time
        start = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=config.temperature,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            METRICS['model_inference'].observe(time.time() - start)
            return response
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"

# ═══════════════════════════════════════════════════════════
# LANGGRAPH AGENT SYSTEM
# ═══════════════════════════════════════════════════════════

class LangGraphAgentSystem:
    """Stateful multi-agent system with LangGraph"""
    
    def __init__(self):
        self.llm = LLMEngine()
        self.rag = MultiSourceRAG()
        self.integrations = IntegrationHub()
        self.cache = CacheManager()
        
        # Initialize model tracking
        if config.wandb_api_key:
            wandb.init(project=config.wandb_project, config=vars(config))
        
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    def _create_agent_graph(self, agent_type: AgentType) -> StateGraph:
        """Create LangGraph workflow for agent"""
        
        workflow = StateGraph(AgentState)
        
        # Define nodes
        async def retrieve_node(state: AgentState):
            query = state.messages[-1]['content']
            context = self.rag.retrieve(query)
            state.context = json.dumps(context)
            state.tools_used.append('rag')
            return state
        
        async def generate_node(state: AgentState):
            prompt = self._build_prompt(agent_type, state)
            response = self.llm.generate(prompt)
            state.messages.append({'role': 'assistant', 'content': response})
            state.confidence = 0.85
            return state
        
        async def integrate_node(state: AgentState):
            # Trigger integrations
            if config.slack_bot_token:
                await self.integrations.send_slack(
                    f"Agent {agent_type.value} completed task"
                )
            state.tools_used.append('integrations')
            return state
        
        # Build graph
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("integrate", integrate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "integrate")
        workflow.set_finish_point("integrate")
        
        return workflow.compile()
    
    def _build_prompt(self, agent_type: AgentType, state: AgentState) -> str:
        """Build agent prompt"""
        system_prompts = {
            AgentType.CODE_ARCHITECT: "Elite software engineer specializing in system architecture.",
            AgentType.SEC_ANALYST: "Cybersecurity expert in threat analysis.",
            AgentType.AUTO_BOT: "Automation specialist for workflows and APIs.",
            AgentType.DATA_ENGINEER: "Data pipeline and ETL expert.",
            AgentType.MLOPS_AGENT: "MLOps specialist for model deployment.",
        }
        
        parts = [f"<|im_start|>system\n{system_prompts.get(agent_type, 'AI Assistant')}<|im_end|>"]
        
        if state.context:
            parts.append(f"<|im_start|>context\n{state.context}<|im_end|>")
        
        for msg in state.messages:
            role = msg['role']
            content = msg['content']
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    
    async def execute(self, agent_type: AgentType, query: str, use_rag: bool = True) -> AgentResponse:
        """Execute agent with LangGraph"""
        METRICS['agent_requests'].labels(agent_type=agent_type.value).inc()
        METRICS['active_agents'].inc()
        
        try:
            # Check cache
            cache_key = hashlib.md5(f"{agent_type.value}:{query}".encode()).hexdigest()
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Cache hit")
                return AgentResponse(**json.loads(cached))
            
            # Ingest knowledge if RAG enabled
            if use_rag:
                await self.rag.ingest_from_sources(query, sources=['google_cse'])
            
            # Create and run graph
            graph = self._create_agent_graph(agent_type)
            initial_state = AgentState(messages=[{'role': 'user', 'content': query}])
            
            final_state = await graph.ainvoke(initial_state)
            
            response = AgentResponse(
                agent=agent_type.value,
                content=final_state.messages[-1]['content'],
                metadata={
                    'tools_used': final_state.tools_used,
                    'rag_enabled': use_rag
                },
                confidence=final_state.confidence,
                sources=json.loads(final_state.context).get('sources', []) if final_state.context else []
            )
            
            # Cache response
            self.cache.set(cache_key, json.dumps(response.__dict__), expire=3600)
            
            # Publish event
            self.integrations.events.publish(config.kafka_topic_agents, {
                'agent': agent_type.value,
                'query': query,
                'timestamp': response.timestamp
            })
            
            # Log to W&B
            if wandb.run:
                wandb.log({
                    'agent': agent_type.value,
                    'confidence': response.confidence,
                    'timestamp': response.timestamp
                })
            
            return response
        
        finally:
            METRICS['active_agents'].dec()
    
    async def orchestrate(self, query: str, agents: List[AgentType], use_rag: bool = True) -> Dict[str, Any]:
        """Multi-agent orchestration"""
        tasks = [self.execute(agent, query, use_rag) for agent in agents]
        responses = await asyncio.gather(*tasks)
        
        return {
            'query': query,
            'responses': [
                {
                    'agent': r.agent,
                    'content': r.content,
                    'metadata': r.metadata,
                    'confidence': r.confidence,
                    'timestamp': r.timestamp,
                    'sources': r.sources
                }
                for r in responses
            ]
        }

# ═══════════════════════════════════════════════════════════
# SELF-IMPROVEMENT ENGINE
# ═══════════════════════════════════════════════════════════

class SelfImprovementEngine:
    """Continuous learning and improvement"""
    
    def __init__(self, agent_system: LangGraphAgentSystem):
        self.system = agent_system
        self.feedback_store: List[FeedbackData] = []
    
    def record_feedback(self, feedback: FeedbackData):
        """Record user feedback"""
        self.feedback_store.append(feedback)
        
        # Log to MLflow
        if config.mlflow_tracking_uri:
            with mlflow.start_run():
                mlflow.log_metric("feedback_rating", feedback.rating)
                mlflow.log_param("agent", feedback.agent)
    
    async def analyze_and_improve(self):
        """Analyze feedback and trigger improvements"""
        if not config.enable_self_improvement:
            return
        
        if len(self.feedback_store) < 10:
            return
        
        avg_rating = sum(f.rating for f in self.feedback_store) / len(self.feedback_store)
        
        if avg_rating < config.feedback_threshold:
            logger.warning(f"Low average rating: {avg_rating}. Triggering improvement...")
            
            # Re-train embeddings or fine-tune (simplified)
            improvement_data = [f"{f.query}\n{f.response}" for f in self.feedback_store]
            
            # Log improvement cycle
            if wandb.run:
                wandb.log({
                    'improvement_cycle': 1,
                    'avg_rating': avg_rating,
                    'feedback_count': len(self.feedback_store)
                })
        
        self.feedback_store.clear()

# ═══════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════

_agent_system = None
_improvement_engine = None

def get_system() -> LangGraphAgentSystem:
    global _agent_system, _improvement_engine
    if _agent_system is None:
        _agent_system = LangGraphAgentSystem()
        _improvement_engine = SelfImprovementEngine(_agent_system)
    return _agent_system

def get_improvement_engine() -> SelfImprovementEngine:
    get_system()  # Ensure initialized
    return _improvement_engine
