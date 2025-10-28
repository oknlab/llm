"""
Unified System - API + Dashboard served from single port.
All access through one NGROK URL.
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
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False

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
            embedding = hashlib.md5(text.encode()).hexdigest()
            self.documents.append(text)
            self.embeddings.append(embedding)
    
    def similarity_search(self, query, k=3):
        import random
        results = []
        for doc in random.sample(self.documents[:k] if len(self.documents) > k else self.documents, 
                                min(k, len(self.documents))):
            results.append(type('Document', (), {'page_content': doc})())
        return results

class AICore:
    """Central orchestrator for multi-agent system."""
    
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
            print(f"🔄 Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("📡 Running in API-only mode")
            self.model = None
            self.tokenizer = None
        
        # Initialize vector store
        if VECTOR_STORE_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                dimension = 384
                self.vector_index = faiss.IndexFlatL2(dimension)
                self.vector_documents = []
                print("✅ Vector store initialized with FAISS")
            except Exception as e:
                print(f"⚠️ FAISS initialization failed: {e}")
                self.vector_store = SimpleVectorStore()
        else:
            self.vector_store = SimpleVectorStore()
            print("📦 Using simple vector store")
        
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
        
        print(f"🤖 Spawned {len(default_configs)} default agents")
    
    def spawn_agent(self, config: AgentConfig) -> 'Agent':
        """Create new agent instance."""
        agent = Agent(config, self)
        self.agents[config.agent_id] = agent
        return agent
    
    def add_to_vector_store(self, texts: List[str]):
        """Add texts to vector store."""
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
        """Live web scraping."""
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
                
                if results:
                    texts = [r['content'] for r in results]
                    self.add_to_vector_store(texts)
                
                return results
                
            except Exception as e:
                print(f"Web scraping error: {e}")
                return []
    
    def generate_response(self, prompt: str, agent_id: str) -> str:
        """Generate response."""
        agent = self.agents.get(agent_id)
        if not agent:
            return "Agent not found"
        
        if not self.model or not self.tokenizer:
            return f"[{agent.config.agent_type.value}] Processing: {prompt[:50]}... (Model offline - using mock response)"
        
        try:
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
            
            agent.update_memory(prompt, response)
            
            return response
            
        except Exception as e:
            return f"[{agent.config.agent_type.value}] Error: {str(e)}"

class Agent:
    """Individual AI agent."""
    
    def __init__(self, config: AgentConfig, core: AICore):
        self.config = config
        self.core = core
        
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
        """Execute assigned task."""
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
            self.performance_metrics['success_rate'] *= 0.95
            return {
                "task_id": task.get('id'),
                "status": "failed",
                "error": str(e)
            }

# FastAPI Application with integrated dashboard
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
    print("✅ AI Core initialized successfully")

# API Models
class TaskRequest(BaseModel):
    agent_id: str
    task_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class AgentCreateRequest(BaseModel):
    agent_type: str
    capabilities: List[str]
    config_overrides: Dict[str, Any] = Field(default_factory=dict)

# Dashboard HTML (embedded)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agents Platform - Unified Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #1a1a2e;
            --secondary: #16213e;
            --accent: #0f3460;
            --highlight: #e94560;
            --text: #f1f1f1;
            --success: #4caf50;
            --warning: #ff9800;
            --error: #f44336;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: rgba(15, 52, 96, 0.3);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, var(--highlight), #f1f1f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .url-info {
            background: rgba(233, 69, 96, 0.2);
            padding: 10px 20px;
            border-radius: 5px;
            margin-top: 15px;
            border: 1px solid var(--highlight);
        }
        
        .url-info code {
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 8px;
            border-radius: 3px;
            font-family: monospace;
        }
        
        .connection-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9rem;
            margin-top: 10px;
        }
        
        .status-online { background: var(--success); }
        .status-offline { background: var(--error); }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(22, 33, 62, 0.5);
            backdrop-filter: blur(5px);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--highlight);
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }
        
        .agents-section, .tasks-section {
            background: rgba(15, 52, 96, 0.2);
            backdrop-filter: blur(5px);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .section-title {
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--highlight);
        }
        
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .agent-card {
            background: rgba(26, 26, 46, 0.6);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid var(--highlight);
            transition: all 0.3s;
        }
        
        .agent-card:hover {
            background: rgba(26, 26, 46, 0.8);
        }
        
        .agent-type {
            font-weight: bold;
            color: var(--highlight);
            margin-bottom: 5px;
        }
        
        .agent-capabilities {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        
        .capability-badge {
            background: var(--accent);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        
        .control-panel {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .btn {
            background: var(--highlight);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: #c13651;
            transform: scale(1.05);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: var(--accent);
        }
        
        .btn-secondary:hover {
            background: #0a2340;
        }
        
        input, select, textarea {
            background: rgba(22, 33, 62, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text);
            padding: 10px;
            border-radius: 5px;
            font-size: 1rem;
            width: 100%;
            margin-bottom: 10px;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--highlight);
        }
        
        .task-form {
            display: grid;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .task-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .task-item {
            background: rgba(26, 26, 46, 0.4);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-badge {
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        
        .status-completed { background: var(--success); }
        .status-pending { background: var(--warning); }
        .status-failed { background: var(--error); }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--highlight);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            animation: slideIn 0.3s ease;
            z-index: 1000;
        }
        
        .notification.error {
            background: var(--error);
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 999;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: var(--secondary);
            padding: 30px;
            border-radius: 10px;
            max-width: 500px;
            width: 90%;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI Agents Platform</h1>
            <p>Enterprise-Grade Multi-Agent Orchestration System</p>
            <div class="url-info">
                📡 Single URL Access: <code id="currentUrl">Loading...</code>
            </div>
            <span class="connection-status" id="connectionStatus">Connecting...</span>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="activeAgents">0</div>
                <div class="stat-label">Active Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="tasksCompleted">0</div>
                <div class="stat-label">Tasks Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgResponseTime">0ms</div>
                <div class="stat-label">Avg Response Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="systemHealth">Checking...</div>
                <div class="stat-label">System Status</div>
            </div>
        </div>
        
        <div class="control-panel">
            <button class="btn" onclick="showCreateAgentModal()">Create Agent</button>
            <button class="btn btn-secondary" onclick="refreshAgents()">Refresh</button>
            <button class="btn btn-secondary" onclick="ingestData()">Ingest Data</button>
            <button class="btn btn-secondary" onclick="window.location.href='/docs'">API Docs</button>
        </div>
        
        <div class="agents-section">
            <h2 class="section-title">Active Agents</h2>
            <div class="agent-grid" id="agentGrid">
                <div class="loading"></div>
            </div>
        </div>
        
        <div class="tasks-section">
            <h2 class="section-title">Task Management</h2>
            
            <div class="task-form">
                <select id="agentSelect">
                    <option value="">Select Agent</option>
                </select>
                <select id="taskType">
                    <option value="generate">Generate</option>
                    <option value="analyze">Analyze</option>
                    <option value="scrape_and_process">Scrape & Process</option>
                </select>
                <textarea id="taskPayload" placeholder="Enter task details..." rows="3"></textarea>
                <button class="btn" onclick="submitTask()">Submit Task</button>
            </div>
            
            <div class="task-list" id="taskList"></div>
        </div>
    </div>
    
    <!-- Create Agent Modal -->
    <div id="createAgentModal" class="modal">
        <div class="modal-content">
            <h2>Create New Agent</h2>
            <br>
            <select id="newAgentType">
                <option value="CODE_ARCHITECT">Code Architect</option>
                <option value="SEC_ANALYST">Security Analyst</option>
                <option value="AUTO_BOT">Automation Bot</option>
                <option value="AGENT_SUITE">Agent Suite</option>
                <option value="CREATIVE_AGENT">Creative Agent</option>
            </select>
            <input type="text" id="newAgentCapabilities" placeholder="Capabilities (comma-separated)">
            <br><br>
            <button class="btn" onclick="createAgent()">Create</button>
            <button class="btn btn-secondary" onclick="hideCreateAgentModal()">Cancel</button>
        </div>
    </div>
    
    <script>
        // Use same origin for API calls (unified service)
        const API_BASE = '';
        let agents = [];
        let tasks = [];
        let isConnected = false;
        
        // Display current URL
        document.getElementById('currentUrl').textContent = window.location.origin;
        
        async function fetchAPI(endpoint, options = {}) {
            try {
                const response = await fetch(`${API_BASE}${endpoint}`, {
                    ...options,
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (!isConnected) {
                    isConnected = true;
                    updateConnectionStatus(true);
                }
                
                return data;
            } catch (error) {
                console.error('API Error:', error);
                updateConnectionStatus(false);
                showNotification(`API Error: ${error.message}`, 'error');
                return null;
            }
        }
        
        function updateConnectionStatus(connected) {
            const status = document.getElementById('connectionStatus');
            if (connected) {
                status.textContent = '✅ Connected';
                status.className = 'connection-status status-online';
            } else {
                status.textContent = '❌ Disconnected';
                status.className = 'connection-status status-offline';
            }
        }
        
        async function refreshAgents() {
            const data = await fetchAPI('/agents');
            if (data) {
                agents = data.agents;
                updateAgentGrid();
                updateAgentSelect();
                updateStats();
            }
        }
        
        function updateAgentGrid() {
            const grid = document.getElementById('agentGrid');
            
            if (agents.length === 0) {
                grid.innerHTML = '<p>No agents active. Create one to get started.</p>';
                return;
            }
            
            grid.innerHTML = agents.map(agent => `
                <div class="agent-card">
                    <div class="agent-type">${agent.type}</div>
                    <div>ID: ${agent.id}</div>
                    <div>Tasks: ${agent.metrics.tasks_completed}</div>
                    <div>Success Rate: ${(agent.metrics.success_rate * 100).toFixed(1)}%</div>
                    <div>Avg Time: ${agent.metrics.avg_response_time.toFixed(2)}s</div>
                    <div class="agent-capabilities">
                        ${agent.capabilities.map(cap => 
                            `<span class="capability-badge">${cap}</span>`
                        ).join('')}
                    </div>
                </div>
            `).join('');
        }
        
        function updateAgentSelect() {
            const select = document.getElementById('agentSelect');
            select.innerHTML = '<option value="">Select Agent</option>' +
                agents.map(agent => 
                    `<option value="${agent.id}">${agent.type} - ${agent.id}</option>`
                ).join('');
        }
        
        function updateStats() {
            document.getElementById('activeAgents').textContent = agents.length;
            const totalTasks = agents.reduce((sum, agent) => 
                sum + agent.metrics.tasks_completed, 0
            );
            document.getElementById('tasksCompleted').textContent = totalTasks;
            
            const avgTime = agents.length > 0 ? agents.reduce((sum, agent) => 
                sum + agent.metrics.avg_response_time, 0
            ) / agents.length : 0;
            document.getElementById('avgResponseTime').textContent = 
                `${(avgTime * 1000).toFixed(0)}ms`;
        }
        
        function showCreateAgentModal() {
            document.getElementById('createAgentModal').style.display = 'flex';
        }
        
        function hideCreateAgentModal() {
            document.getElementById('createAgentModal').style.display = 'none';
        }
        
        async function createAgent() {
            const agentType = document.getElementById('newAgentType').value;
            const capabilities = document.getElementById('newAgentCapabilities').value;
            
            if (!capabilities) {
                showNotification('Please enter capabilities', 'error');
                return;
            }
            
            const data = await fetchAPI('/agents/create', {
                method: 'POST',
                body: JSON.stringify({
                    agent_type: agentType,
                    capabilities: capabilities.split(',').map(c => c.trim()),
                    config_overrides: {}
                })
            });
            
            if (data) {
                showNotification('Agent created successfully');
                hideCreateAgentModal();
                document.getElementById('newAgentCapabilities').value = '';
                refreshAgents();
            }
        }
        
        async function submitTask() {
            const agentId = document.getElementById('agentSelect').value;
            const taskType = document.getElementById('taskType').value;
            const payload = document.getElementById('taskPayload').value;
            
            if (!agentId || !payload) {
                showNotification('Please select agent and enter task details', 'error');
                return;
            }
            
            const data = await fetchAPI('/tasks/submit', {
                method: 'POST',
                body: JSON.stringify({
                    agent_id: agentId,
                    task_type: taskType,
                    payload: taskType === 'generate' ? 
                        { prompt: payload } : 
                        { query: payload }
                })
            });
            
            if (data) {
                addTaskToList(data);
                document.getElementById('taskPayload').value = '';
                showNotification('Task submitted');
            }
        }
        
        function addTaskToList(task) {
            const taskList = document.getElementById('taskList');
            const taskElement = document.createElement('div');
            taskElement.className = 'task-item';
            
            let resultText = '';
            if (task.result) {
                if (typeof task.result === 'string') {
                    resultText = task.result.substring(0, 100) + '...';
                } else {
                    resultText = JSON.stringify(task.result).substring(0, 100) + '...';
                }
            }
            
            taskElement.innerHTML = `
                <div>
                    <strong>Task ${task.task_id}</strong>
                    <div>Time: ${task.execution_time?.toFixed(2)}s</div>
                    ${resultText ? `<div style="font-size: 0.9rem; opacity: 0.8; margin-top: 5px;">${resultText}</div>` : ''}
                </div>
                <span class="status-badge status-${task.status}">
                    ${task.status}
                </span>
            `;
            taskList.insertBefore(taskElement, taskList.firstChild);
            
            if (taskList.children.length > 10) {
                taskList.removeChild(taskList.lastChild);
            }
        }
        
        async function ingestData() {
            const query = prompt('Enter search query for data ingestion:');
            if (query) {
                const data = await fetchAPI(`/rag/ingest?query=${encodeURIComponent(query)}`, {
                    method: 'POST'
                });
                
                if (data) {
                    showNotification(`Ingested ${data.documents_ingested} documents`);
                }
            }
        }
        
        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = type === 'error' ? 'notification error' : 'notification';
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }
        
        async function checkHealth() {
            const data = await fetchAPI('/health');
            if (data) {
                const healthStatus = document.getElementById('systemHealth');
                healthStatus.textContent = data.status === 'operational' ? '✅ Online' : '⚠️ Issues';
                
                if (data.model_loaded === false) {
                    healthStatus.textContent += ' (API)';
                }
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            refreshAgents();
            checkHealth();
            
            // Periodic updates
            setInterval(refreshAgents, 30000);
            setInterval(checkHealth, 60000);
        });
    </script>
</body>
</html>
"""

# Serve Dashboard at root
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the unified dashboard at root URL."""
    return DASHBOARD_HTML

# Also serve at /dashboard for clarity
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard_alt():
    """Alternative dashboard endpoint."""
    return DASHBOARD_HTML

# API Endpoints
@app.post("/agents/create")
async def create_agent(request: AgentCreateRequest):
    """Create new custom agent."""
    try:
        agent_id = f"{request.agent_type}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('UNIFIED_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
