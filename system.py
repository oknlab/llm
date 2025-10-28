"""
Core AI Agents System - Production ready with fallback mechanisms.
Implements agentic RAG with modular architecture.
"""

import os
import json
import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Conditional imports with fallbacks
try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available, using fallback implementation")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("Warning: Vector store dependencies not available, using simple storage")

import httpx
from bs4 import BeautifulSoup

# Agent Types
class AgentType(Enum):
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"  
    AUTO_BOT = "AutoBot"
    AGENT_SUITE = "AgentSuite"
    CREATIVE_AGENT = "CreativeAgent"

@dataclass
class AgentConfig:
    """Configuration for individual AI agents."""
    agent_id: str
    agent_type: AgentType
    capabilities: List[str]
    model_config: Dict[str, Any]
    memory_window: int = 10
    temperature: float = 0.7
    max_tokens: int = 2048

# Fallback implementations
class SimpleMemory:
    """Simple memory implementation when LangChain not available."""
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size * 2)
    
    def save_context(self, input_dict, output_dict):
        self.messages.append({"role": "user", "content": input_dict.get("input", "")})
        self.messages.append({"role": "assistant", "content": output_dict.get("output", "")})
    
    def get_messages(self):
        return list(self.messages)

class SimpleVectorStore:
    """Simple vector store implementation when FAISS not available."""
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def add_texts(self, texts):
        for text in texts:
            # Simple hash-based "embedding"
            embedding = hashlib.md5(text.encode()).hexdigest()
            self.documents.append(text)
            self.embeddings.append(embedding)
    
    def similarity_search(self, query, k=3):
        # Return random documents as fallback
        import random
        results = []
        for doc in random.sample(self.documents[:k] if len(self.documents) > k else self.documents, 
                                min(k, len(self.documents))):
            results.append(type('Document', (), {'page_content': doc})())
        return results

class AICore:
    """Central orchestrator for multi-agent system with fallback support."""
    
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct')
        self.api_base = os.getenv('MODEL_SERVER', 'http://localhost:8001')
        self.api_key = os.getenv('API_KEY', 'dummy-key')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.agents: Dict[str, 'Agent'] = {}
        self.vector_store = None
        self.embeddings_model = None
        self.tokenizer = None
        self.model = None
        
        self._initialize_core()
    
    def _initialize_core(self):
        """Initialize core components with fallback options."""
        try:
            # Load tokenizer and model
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Use smaller model for Colab compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Running in API-only mode")
            self.model = None
            self.tokenizer = None
        
        # Initialize vector store
        if VECTOR_STORE_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                # Initialize FAISS index
                dimension = 384  # all-MiniLM-L6-v2 dimension
                self.vector_index = faiss.IndexFlatL2(dimension)
                self.vector_documents = []
                print("Vector store initialized with FAISS")
            except Exception as e:
                print(f"FAISS initialization failed: {e}")
                self.vector_store = SimpleVectorStore()
        else:
            self.vector_store = SimpleVectorStore()
            print("Using simple vector store")
        
        # Initialize default agents
        self._spawn_default_agents()
    
    def _spawn_default_agents(self):
        """Create default agent instances."""
        default_configs = [
            AgentConfig(
                agent_id="code_001",
                agent_type=AgentType.CODE_ARCHITECT,
                capabilities=["python", "javascript", "rust", "api_design"],
                model_config={"temperature": 0.3}
            ),
            AgentConfig(
                agent_id="sec_001",
                agent_type=AgentType.SEC_ANALYST,
                capabilities=["penetration_testing", "audit", "threat_modeling"],
                model_config={"temperature": 0.2}
            ),
            AgentConfig(
                agent_id="auto_001",
                agent_type=AgentType.AUTO_BOT,
                capabilities=["workflow_automation", "api_integration"],
                model_config={"temperature": 0.5}
            ),
            AgentConfig(
                agent_id="suite_001",
                agent_type=AgentType.AGENT_SUITE,
                capabilities=["reporting", "finance", "operations"],
                model_config={"temperature": 0.4}
            ),
            AgentConfig(
                agent_id="creative_001",
                agent_type=AgentType.CREATIVE_AGENT,
                capabilities=["content_generation", "audio", "visual"],
                model_config={"temperature": 0.8}
            )
        ]
        
        for config in default_configs:
            self.spawn_agent(config)
    
    def spawn_agent(self, config: AgentConfig) -> 'Agent':
        """Create new agent instance."""
        agent = Agent(config, self)
        self.agents[config.agent_id] = agent
        return agent
    
    def add_to_vector_store(self, texts: List[str]):
        """Add texts to vector store with appropriate method."""
        if VECTOR_STORE_AVAILABLE and hasattr(self, 'embeddings_model'):
            embeddings = self.embeddings_model.encode(texts)
            self.vector_index.add(np.array(embeddings).astype('float32'))
            self.vector_documents.extend(texts)
        elif self.vector_store:
            self.vector_store.add_texts(texts)
    
    def search_vector_store(self, query: str, k: int = 3) -> List[str]:
        """Search vector store for similar documents."""
        if VECTOR_STORE_AVAILABLE and hasattr(self, 'embeddings_model'):
            query_embedding = self.embeddings_model.encode([query])
            distances, indices = self.vector_index.search(
                np.array(query_embedding).astype('float32'), k
            )
            return [self.vector_documents[i] for i in indices[0] if i < len(self.vector_documents)]
        elif self.vector_store:
            docs = self.vector_store.similarity_search(query, k)
            return [doc.page_content for doc in docs]
        return []
    
    async def web_scrape(self, query: str) -> List[Dict]:
        """Live web scraping with error handling."""
        cse_url = f"https://www.google.com/search?q={query}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    cse_url, 
                    timeout=10.0,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                # Parse Google search results
                for g in soup.find_all('div', class_='g')[:5]:
                    title_elem = g.find('h3')
                    snippet_elem = g.find('span', class_='aCOpRe')
                    
                    if title_elem:
                        result = {
                            'title': title_elem.text,
                            'snippet': snippet_elem.text if snippet_elem else '',
                            'content': f"{title_elem.text}\n{snippet_elem.text if snippet_elem else ''}"
                        }
                        results.append(result)
                
                # Add to vector store
                if results:
                    texts = [r['content'] for r in results]
                    self.add_to_vector_store(texts)
                
                return results
                
            except Exception as e:
                print(f"Web scraping error: {e}")
                return []
    
    def generate_response(self, prompt: str, agent_id: str) -> str:
        """Generate response with fallback to simple generation."""
        agent = self.agents.get(agent_id)
        if not agent:
            return "Agent not found"
        
        # If model not available, return mock response
        if not self.model or not self.tokenizer:
            return f"[{agent.config.agent_type.value}] Processing: {prompt[:50]}... (Model offline - using mock response)"
        
        try:
            # Apply agent-specific prompt engineering
            context = agent.get_context()
            full_prompt = f"{context}\n\nUser: {prompt}\nAgent:"
            
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(agent.config.max_tokens, 256),
                    temperature=agent.config.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Agent:")[-1].strip()
            
            # Update agent memory
            agent.update_memory(prompt, response)
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"[{agent.config.agent_type.value}] Error processing request: {str(e)}"

class Agent:
    """Individual AI agent with specialized capabilities."""
    
    def __init__(self, config: AgentConfig, core: AICore):
        self.config = config
        self.core = core
        
        # Use appropriate memory implementation
        if LANGCHAIN_AVAILABLE:
            self.memory = ConversationBufferWindowMemory(
                k=config.memory_window,
                return_messages=True
            )
        else:
            self.memory = SimpleMemory(config.memory_window)
        
        self.task_queue: List[Dict] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 1.0,
            "avg_response_time": 0.0
        }
    
    def get_context(self) -> str:
        """Build agent-specific context."""
        context_parts = [
            f"You are {self.config.agent_type.value}.",
            f"Capabilities: {', '.join(self.config.capabilities)}.",
            "Provide precise, actionable responses."
        ]
        
        # Add specialized instructions
        context_map = {
            AgentType.CODE_ARCHITECT: "Focus on clean, optimized code with best practices.",
            AgentType.SEC_ANALYST: "Prioritize security vulnerabilities and threat mitigation.",
            AgentType.AUTO_BOT: "Design efficient automation workflows.",
            AgentType.AGENT_SUITE: "Provide data-driven insights and operational efficiency.",
            AgentType.CREATIVE_AGENT: "Generate engaging, original content."
        }
        
        if self.config.agent_type in context_map:
            context_parts.append(context_map[self.config.agent_type])
        
        return "\n".join(context_parts)
    
    def update_memory(self, input_text: str, output_text: str):
        """Update agent conversation memory."""
        self.memory.save_context({"input": input_text}, {"output": output_text})
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task with timing."""
        start_time = time.time()
        
        try:
            task_type = task.get('type')
            payload = task.get('payload', {})
            
            if task_type == 'generate':
                result = self.core.generate_response(
                    payload.get('prompt', ''),
                    self.config.agent_id
                )
            elif task_type == 'analyze':
                # Retrieve relevant context
                docs = self.core.search_vector_store(
                    payload.get('query', ''),
                    k=3
                )
                context = "\n".join(docs)
                result = self.core.generate_response(
                    f"Context: {context}\n\nAnalyze: {payload.get('query', '')}",
                    self.config.agent_id
                )
            elif task_type == 'scrape_and_process':
                docs = await self.core.web_scrape(payload.get('query', ''))
                result = {
                    "documents": len(docs),
                    "content": [d.get('content', '')[:200] for d in docs]
                }
            else:
                result = {"error": "Unknown task type"}
            
            # Update metrics
            elapsed = time.time() - start_time
            self.performance_metrics['tasks_completed'] += 1
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * 
                 (self.performance_metrics['tasks_completed'] - 1) + elapsed) /
                self.performance_metrics['tasks_completed']
            )
            
            return {
                "task_id": task.get('id'),
                "status": "completed",
                "result": result,
                "execution_time": elapsed
            }
            
        except Exception as e:
            self.performance_metrics['success_rate'] *= 0.95  # Decay success rate
            return {
                "task_id": task.get('id'),
                "status": "failed",
                "error": str(e)
            }

# FastAPI Application
app = FastAPI(title="AI Agents Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global AI Core instance
ai_core = None

@app.on_event("startup")
async def startup_event():
    global ai_core
    ai_core = AICore()
    print("AI Core initialized successfully")

# API Models
class TaskRequest(BaseModel):
    agent_id: str
    task_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class AgentCreateRequest(BaseModel):
    agent_type: str
    capabilities: List[str]
    config_overrides: Dict[str, Any] = Field(default_factory=dict)

# API Endpoints
@app.post("/agents/create")
async def create_agent(request: AgentCreateRequest):
    """Create new custom agent."""
    try:
        agent_id = f"{request.agent_type}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        # Validate agent type
        try:
            agent_type_enum = AgentType[request.agent_type.upper().replace(' ', '_')]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid agent type: {request.agent_type}")
        
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type_enum,
            capabilities=request.capabilities,
            model_config=request.config_overrides
        )
        agent = ai_core.spawn_agent(config)
        return {
            "agent_id": agent_id,
            "status": "created",
            "capabilities": agent.config.capabilities
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all active agents."""
    return {
        "agents": [
            {
                "id": agent_id,
                "type": agent.config.agent_type.value,
                "capabilities": agent.config.capabilities,
                "metrics": agent.performance_metrics
            }
            for agent_id, agent in ai_core.agents.items()
        ]
    }

@app.post("/tasks/submit")
async def submit_task(request: TaskRequest):
    """Submit task to specific agent."""
    agent = ai_core.agents.get(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    task = {
        "id": hashlib.md5(f"{request.agent_id}{datetime.now()}".encode()).hexdigest()[:12],
        "type": request.task_type,
        "payload": request.payload
    }
    
    # Execute task
    result = await agent.execute_task(task)
    return result

@app.post("/rag/ingest")
async def ingest_data(query: str):
    """Ingest data through web scraping."""
    docs = await ai_core.web_scrape(query)
    return {
        "documents_ingested": len(docs),
        "status": "success" if docs else "no_results"
    }

@app.get("/health")
async def health_check():
    """System health check."""
    vector_store_size = 0
    if hasattr(ai_core, 'vector_index'):
        vector_store_size = ai_core.vector_index.ntotal if ai_core.vector_index else 0
    elif hasattr(ai_core, 'vector_store') and hasattr(ai_core.vector_store, 'documents'):
        vector_store_size = len(ai_core.vector_store.documents)
    
    return {
        "status": "operational",
        "agents_active": len(ai_core.agents),
        "vector_store_size": vector_store_size,
        "device": ai_core.device,
        "model_loaded": ai_core.model is not None
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Agents Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
