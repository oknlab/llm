"""
Elite AI Agent System - Core Engine
Architecture: Modular, Scalable, Production-Grade
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from pydantic_settings import BaseSettings
from pydantic import Field

# ═══════════════════════════════════════════════════════════
# CONFIGURATION MANAGEMENT
# ═══════════════════════════════════════════════════════════

class Settings(BaseSettings):
    ngrok_auth_token: str = Field(default="")
    model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    model_server: str = Field(default="local")
    api_key: str = Field(default="none")
    device: str = Field(default="cuda")
    max_length: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store: str = Field(default="chroma")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)
    cse_url: str = Field(default="")
    max_scrape_pages: int = Field(default=10)
    request_timeout: int = Field(default=30)
    user_agent: str = Field(default="")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    max_workers: int = Field(default=4)

    class Config:
        env_file = ".env"
        case_sensitive = False


config = Settings()

# ═══════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# AGENT TYPES
# ═══════════════════════════════════════════════════════════

class AgentType(Enum):
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"
    AUTO_BOT = "AutoBot"
    AG_CUSTOM = "AgCustom"

# ═══════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    agent: str
    content: str
    metadata: Dict[str, Any]
    confidence: float

@dataclass
class RAGContext:
    documents: List[Document]
    sources: List[str]
    relevance_scores: List[float]

# ═══════════════════════════════════════════════════════════
# WEB SCRAPING ENGINE
# ═══════════════════════════════════════════════════════════

class WebScraper:
    """Production-grade web scraping with anti-detection"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def scrape_url(self, url: str) -> Optional[str]:
        """Scrape single URL with error handling"""
        try:
            response = self.session.get(
                url,
                headers=self._get_headers(),
                timeout=config.request_timeout,
                verify=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script, style, nav, footer
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            return text
        
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            return None
    
    def search_and_scrape(self, query: str, max_results: int = None) -> List[Document]:
        """Search CSE and scrape results"""
        max_results = max_results or config.max_scrape_pages
        documents = []
        
        try:
            # Build search query
            search_url = f"{config.cse_url}&q={requests.utils.quote(query)}"
            response = self.session.get(search_url, headers=self._get_headers(), timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            links = []
            
            # Extract search result links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http') and 'google' not in href:
                    links.append(href)
                if len(links) >= max_results:
                    break
            
            # Scrape each URL in parallel
            futures = [self.executor.submit(self.scrape_url, url) for url in links[:max_results]]
            
            for i, future in enumerate(futures):
                content = future.result()
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': links[i],
                            'query': query,
                            'hash': hashlib.md5(content.encode()).hexdigest()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Scraped {len(documents)} documents for query: {query}")
            return documents
        
        except Exception as e:
            logger.error(f"Search and scrape failed: {e}")
            return []

# ═══════════════════════════════════════════════════════════
# RAG ENGINE
# ═══════════════════════════════════════════════════════════

class RAGEngine:
    """Advanced Retrieval-Augmented Generation Engine"""
    
    def __init__(self):
        logger.info("Initializing RAG Engine...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.device if torch.cuda.is_available() else 'cpu'}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.scraper = WebScraper()
        
        logger.info("RAG Engine initialized successfully")
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """Ingest and index documents"""
        if not documents:
            logger.warning("No documents to ingest")
            return
        
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="rag_collection"
            )
        else:
            self.vectorstore.add_documents(splits)
        
        logger.info(f"Indexed {len(splits)} chunks")
    
    def retrieve_context(self, query: str, top_k: int = None) -> RAGContext:
        """Retrieve relevant context for query"""
        top_k = top_k or config.top_k
        
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return RAGContext(documents=[], sources=[], relevance_scores=[])
        
        # Retrieve documents with scores
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        documents = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        sources = [doc.metadata.get('source', 'unknown') for doc in documents]
        
        return RAGContext(
            documents=documents,
            sources=sources,
            relevance_scores=scores
        )

# ═══════════════════════════════════════════════════════════
# LLM ENGINE
# ═══════════════════════════════════════════════════════════

class LLMEngine:
    """Qwen3-1.7B Local Inference Engine"""
    
    def __init__(self):
        logger.info(f"Loading model: {config.model_name}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Quantization config for memory efficiency
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only new content
            response = response[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"

# ═══════════════════════════════════════════════════════════
# AGENT SYSTEM
# ═══════════════════════════════════════════════════════════

class AgentOrchestrator:
    """Multi-Agent Orchestration System"""
    
    def __init__(self):
        self.llm = LLMEngine()
        self.rag = RAGEngine()
        
        # Agent system prompts
        self.agent_prompts = {
            AgentType.CODE_ARCHITECT: """You are CodeArchitect, an elite software engineer specializing in complex systems design, API development, and code architecture. You write production-grade code in Python, JavaScript, Rust, and Go. Provide precise, executable solutions.""",
            
            AgentType.SEC_ANALYST: """You are SecAnalyst, a cybersecurity expert specializing in penetration testing, security audits, and threat modeling. You identify vulnerabilities and provide actionable security recommendations.""",
            
            AgentType.AUTO_BOT: """You are AutoBot, an automation specialist. You design and implement workflows, API integrations, and automation pipelines using tools like FastAPI, n8n, and various APIs.""",
            
            AgentType.AG_CUSTOM: """You are AgCustom, a versatile AI agent builder. You design custom AI agents tailored to specific use cases, combining various AI capabilities and tools."""
        }
    
    def _build_prompt(self, agent_type: AgentType, query: str, context: Optional[RAGContext] = None) -> str:
        """Build agent-specific prompt with RAG context"""
        system_prompt = self.agent_prompts[agent_type]
        
        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        
        if context and context.documents:
            context_text = "\n\n".join([
                f"[Source {i+1}]: {doc.page_content[:500]}"
                for i, doc in enumerate(context.documents[:3])
            ])
            prompt_parts.append(f"<|im_start|>context\n{context_text}<|im_end|>")
        
        prompt_parts.append(f"<|im_start|>user\n{query}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    async def execute_agent(self, agent_type: AgentType, query: str, use_rag: bool = True) -> AgentResponse:
        """Execute agent with optional RAG"""
        logger.info(f"Executing {agent_type.value} for query: {query[:100]}...")
        
        context = None
        if use_rag:
            # Scrape and index new data
            documents = self.rag.scraper.search_and_scrape(query, max_results=5)
            if documents:
                self.rag.ingest_documents(documents)
            
            # Retrieve context
            context = self.rag.retrieve_context(query)
        
        # Build prompt
        prompt = self._build_prompt(agent_type, query, context)
        
        # Generate response
        response = self.llm.generate(prompt, max_new_tokens=1024)
        
        # Calculate confidence (simplified)
        confidence = 0.85 if context and context.documents else 0.60
        
        return AgentResponse(
            agent=agent_type.value,
            content=response,
            metadata={
                'sources': context.sources if context else [],
                'num_context_docs': len(context.documents) if context else 0,
                'rag_enabled': use_rag
            },
            confidence=confidence
        )
    
    async def orchestrate(self, query: str, agents: List[AgentType] = None, use_rag: bool = True) -> Dict[str, Any]:
        """Orchestrate multiple agents"""
        agents = agents or [AgentType.CODE_ARCHITECT]
        
        # Execute agents in parallel
        tasks = [self.execute_agent(agent, query, use_rag) for agent in agents]
        responses = await asyncio.gather(*tasks)
        
        return {
            'query': query,
            'responses': [
                {
                    'agent': r.agent,
                    'content': r.content,
                    'metadata': r.metadata,
                    'confidence': r.confidence
                }
                for r in responses
            ]
        }

# ═══════════════════════════════════════════════════════════
# SYSTEM INITIALIZATION
# ═══════════════════════════════════════════════════════════

_orchestrator = None

def get_orchestrator() -> AgentOrchestrator:
    """Singleton orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
