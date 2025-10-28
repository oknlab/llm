"""
ELITE MULTI-AGENT SYSTEM - CORE ORCHESTRATION ENGINE
Enterprise-grade AI Platform with Agentic RAG & Custom Agent Creation
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Core Imports
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool, BaseTool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from loguru import logger
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from queue import PriorityQueue
import hashlib


# ============================================
# CONFIGURATION MANAGEMENT
# ============================================

class Config:
    """Centralized configuration with validation"""
    
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    MODEL_SERVER: str = os.getenv("MODEL_SERVER", "http://localhost:8000/v1")
    API_KEY: str = os.getenv("API_KEY", "local-qwen-key")
    
    GOOGLE_CSE_API_KEY: str = os.getenv("GOOGLE_CSE_API_KEY", "")
    GOOGLE_CSE_CX: str = os.getenv("GOOGLE_CSE_CX", "014662525286492529401:2upbuo2qpni")
    
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "120"))
    ENABLE_MEMORY: bool = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
    MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "10"))
    
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))


# ============================================
# ADVANCED LLM CLIENT
# ============================================

class LLMClient:
    """Production-grade LLM client with retry logic and error handling"""
    
    def __init__(self):
        self.base_url = Config.MODEL_SERVER
        self.api_key = Config.API_KEY
        self.model = Config.MODEL_NAME
        
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate completion with exponential backoff retry"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens or Config.MAX_TOKENS,
                        "temperature": temperature or Config.TEMPERATURE,
                        "top_p": Config.TEMPERATURE,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=Config.AGENT_TIMEOUT
                )
                response.raise_for_status()
                return response.json()["choices"][0]["text"].strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM generation failed after {max_retries} attempts: {e}")
                    return f"Error: {str(e)}"
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s")
                asyncio.sleep(delay)
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": Config.MAX_TOKENS,
                    "temperature": Config.TEMPERATURE,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=Config.AGENT_TIMEOUT
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return f"Error: {str(e)}"


# ============================================
# LIVE WEB SCRAPING RAG ENGINE
# ============================================

class WebScrapingRAG:
    """Production RAG with live Google Custom Search scraping"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.RAG_CHUNK_SIZE,
            chunk_overlap=Config.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        self.cache = {}
        logger.info("WebScrapingRAG initialized")
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Live Google Custom Search with content extraction"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.cache:
            logger.info(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        
        try:
            # Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": Config.GOOGLE_CSE_API_KEY,
                "cx": Config.GOOGLE_CSE_CX,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            documents = []
            for item in search_results.get("items", []):
                # Extract and clean content
                content = self._extract_content(item.get("link", ""))
                if content:
                    documents.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "content": content
                    })
            
            self.cache[cache_key] = documents
            logger.info(f"Retrieved {len(documents)} documents for: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def _extract_content(self, url: str) -> str:
        """Extract clean text from URL"""
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; AIBot/1.0)'
            })
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines[:100])  # Limit content
            
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
            return ""
    
    def build_vector_store(self, documents: List[Dict[str, str]]) -> Chroma:
        """Build Chroma vector store from documents"""
        texts = []
        metadatas = []
        
        for doc in documents:
            # Combine title, snippet, and content
            full_text = f"{doc['title']}\n{doc['snippet']}\n{doc['content']}"
            chunks = self.text_splitter.split_text(full_text)
            
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({
                    "title": doc["title"],
                    "link": doc["link"],
                    "source": "web_search"
                })
        
        if not texts:
            logger.warning("No texts to build vector store")
            return None
        
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=Config.VECTOR_STORE_PATH
        )
        
        logger.info(f"Vector store built with {len(texts)} chunks")
        return self.vector_store
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        k = top_k or Config.RAG_TOP_K
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            for doc, score in results
        ]
    
    def query(self, question: str) -> str:
        """Full RAG pipeline: search -> retrieve -> format"""
        # Step 1: Web search
        documents = self.search_web(question)
        
        if not documents:
            return "No relevant information found."
        
        # Step 2: Build vector store
        self.build_vector_store(documents)
        
        # Step 3: Retrieve relevant chunks
        relevant_docs = self.retrieve(question)
        
        # Step 4: Format context
        context = "\n\n".join([
            f"[Source: {doc['metadata']['title']}]\n{doc['content']}"
            for doc in relevant_docs
        ])
        
        return context


# ============================================
# BASE AGENT ARCHITECTURE
# ============================================

class AgentCapability(Enum):
    """Agent capability types"""
    CODE = "code"
    SECURITY = "security"
    AUTOMATION = "automation"
    ADMIN = "admin"
    CREATIVE = "creative"
    CUSTOM = "custom"


class AgentState(BaseModel):
    """Shared state across agents"""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    rag_context: str = ""
    current_agent: Optional[str] = None
    iteration: int = 0
    final_output: str = ""


class BaseAgent(ABC):
    """Abstract base agent with core capabilities"""
    
    def __init__(self, name: str, capability: AgentCapability, llm_client: LLMClient):
        self.name = name
        self.capability = capability
        self.llm = llm_client
        self.tools = self._initialize_tools()
        self.memory = ConversationBufferWindowMemory(
            k=Config.MEMORY_WINDOW,
            return_messages=True,
            memory_key="chat_history"
        ) if Config.ENABLE_MEMORY else None
        
        logger.info(f"Agent initialized: {name} ({capability.value})")
    
    @abstractmethod
    def _initialize_tools(self) -> List[Tool]:
        """Initialize agent-specific tools"""
        pass
    
    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """Process agent task"""
        pass
    
    def _execute_with_tools(self, query: str, context: str = "") -> str:
        """Execute query with available tools"""
        prompt = f"""You are {self.name}, an expert in {self.capability.value}.

Context:
{context}

Task: {query}

Use your tools and expertise to provide a comprehensive solution.
"""
        return self.llm.generate(prompt)


# ============================================
# SPECIALIZED AGENT IMPLEMENTATIONS
# ============================================

class CodeArchitect(BaseAgent):
    """Expert in Python, JS, Rust, Go, API development"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("CodeArchitect", AgentCapability.CODE, llm_client)
    
    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="code_generator",
                func=self._generate_code,
                description="Generate production-grade code in Python, JS, Rust, Go"
            ),
            Tool(
                name="code_optimizer",
                func=self._optimize_code,
                description="Optimize code for performance and scalability"
            ),
            Tool(
                name="api_designer",
                func=self._design_api,
                description="Design RESTful and GraphQL APIs"
            )
        ]
    
    def _generate_code(self, requirements: str) -> str:
        prompt = f"""Generate clean, modular, production-ready code for:
{requirements}

Requirements:
- Follow SOLID principles
- Include error handling
- Add type hints and docstrings
- Optimize for performance
"""
        return self.llm.generate(prompt)
    
    def _optimize_code(self, code: str) -> str:
        prompt = f"""Optimize this code for:
- Time complexity
- Space complexity
- Readability
- Scalability

Code:
{code}
"""
        return self.llm.generate(prompt)
    
    def _design_api(self, specs: str) -> str:
        prompt = f"""Design a scalable API with:
- RESTful endpoints
- Authentication
- Rate limiting
- Documentation

Specifications:
{specs}
"""
        return self.llm.generate(prompt)
    
    def process(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Processing code task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        result = self._execute_with_tools(query, state.rag_context)
        
        state.messages.append({
            "role": "assistant",
            "content": result,
            "agent": self.name
        })
        state.current_agent = self.name
        return state


class SecAnalyst(BaseAgent):
    """Security expert: pen testing, audits, threat modeling"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("SecAnalyst", AgentCapability.SECURITY, llm_client)
    
    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="security_audit",
                func=self._audit_security,
                description="Perform comprehensive security audit"
            ),
            Tool(
                name="threat_model",
                func=self._threat_modeling,
                description="Create threat models and mitigation strategies"
            ),
            Tool(
                name="vulnerability_scan",
                func=self._scan_vulnerabilities,
                description="Scan for common vulnerabilities (OWASP Top 10)"
            )
        ]
    
    def _audit_security(self, target: str) -> str:
        prompt = f"""Perform security audit for:
{target}

Check for:
- Authentication/Authorization flaws
- Injection vulnerabilities
- Cryptographic issues
- Configuration errors
- Data exposure risks
"""
        return self.llm.generate(prompt)
    
    def _threat_modeling(self, system: str) -> str:
        prompt = f"""Create threat model using STRIDE methodology for:
{system}

Include:
- Attack surface analysis
- Threat scenarios
- Mitigation strategies
- Security controls
"""
        return self.llm.generate(prompt)
    
    def _scan_vulnerabilities(self, code: str) -> str:
        prompt = f"""Scan for vulnerabilities (OWASP Top 10):
{code}

Identify:
- SQL Injection
- XSS
- CSRF
- Insecure Deserialization
- Security Misconfigurations
"""
        return self.llm.generate(prompt)
    
    def process(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Processing security task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        result = self._execute_with_tools(query, state.rag_context)
        
        state.messages.append({
            "role": "assistant",
            "content": result,
            "agent": self.name
        })
        state.current_agent = self.name
        return state


class AutoBot(BaseAgent):
    """Automation specialist: workflows, API integrations"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("AutoBot", AgentCapability.AUTOMATION, llm_client)
    
    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="workflow_designer",
                func=self._design_workflow,
                description="Design automation workflows"
            ),
            Tool(
                name="api_integrator",
                func=self._integrate_api,
                description="Integrate with external APIs (Zapier, n8n, Make)"
            )
        ]
    
    def _design_workflow(self, requirements: str) -> str:
        prompt = f"""Design automation workflow for:
{requirements}

Include:
- Triggers and actions
- Error handling
- Retry logic
- Monitoring
"""
        return self.llm.generate(prompt)
    
    def _integrate_api(self, api_specs: str) -> str:
        prompt = f"""Create API integration for:
{api_specs}

Include:
- Authentication
- Request/response handling
- Rate limiting
- Error handling
"""
        return self.llm.generate(prompt)
    
    def process(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Processing automation task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        result = self._execute_with_tools(query, state.rag_context)
        
        state.messages.append({
            "role": "assistant",
            "content": result,
            "agent": self.name
        })
        state.current_agent = self.name
        return state


class AgentSuite(BaseAgent):
    """Admin agent: reports, finance, ops automation"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("AgentSuite", AgentCapability.ADMIN, llm_client)
    
    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="report_generator",
                func=self._generate_report,
                description="Generate comprehensive reports"
            ),
            Tool(
                name="ops_automation",
                func=self._automate_ops,
                description="Automate operational tasks"
            )
        ]
    
    def _generate_report(self, data: str) -> str:
        prompt = f"""Generate executive report from:
{data}

Include:
- Summary
- Key metrics
- Insights
- Recommendations
"""
        return self.llm.generate(prompt)
    
    def _automate_ops(self, task: str) -> str:
        prompt = f"""Automate operational task:
{task}

Provide:
- Automation script
- Scheduling strategy
- Monitoring setup
"""
        return self.llm.generate(prompt)
    
    def process(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Processing admin task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        result = self._execute_with_tools(query, state.rag_context)
        
        state.messages.append({
            "role": "assistant",
            "content": result,
            "agent": self.name
        })
        state.current_agent = self.name
        return state


class CreativeAgent(BaseAgent):
    """Content creation: text, audio, visual"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__("CreativeAgent", AgentCapability.CREATIVE, llm_client)
    
    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="content_writer",
                func=self._write_content,
                description="Write engaging content"
            ),
            Tool(
                name="script_generator",
                func=self._generate_script,
                description="Generate scripts for audio/video"
            )
        ]
    
    def _write_content(self, brief: str) -> str:
        prompt = f"""Create engaging content for:
{brief}

Requirements:
- SEO optimized
- Audience-focused
- Clear call-to-action
"""
        return self.llm.generate(prompt)
    
    def _generate_script(self, topic: str) -> str:
        prompt = f"""Generate script for:
{topic}

Include:
- Hook
- Main content
- Conclusion
"""
        return self.llm.generate(prompt)
    
    def process(self, state: AgentState) -> AgentState:
        logger.info(f"[{self.name}] Processing creative task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        result = self._execute_with_tools(query, state.rag_context)
        
        state.messages.append({
            "role": "assistant",
            "content": result,
            "agent": self.name
        })
        state.current_agent = self.name
        return state


# ============================================
# CUSTOM AGENT FACTORY
# ============================================

class CustomAgentFactory:
    """Factory for creating custom AI agents dynamically"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.registry = {}
    
    def create_agent(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        system_prompt: str
    ) -> BaseAgent:
        """Create custom agent with specified capabilities"""
        
        class DynamicAgent(BaseAgent):
            def __init__(self, llm_client: LLMClient):
                self.description = description
                self.capabilities_list = capabilities
                self.system_prompt = system_prompt
                super().__init__(name, AgentCapability.CUSTOM, llm_client)
            
            def _initialize_tools(self) -> List[Tool]:
                tools = []
                for cap in self.capabilities_list:
                    tools.append(Tool(
                        name=f"{cap}_tool",
                        func=lambda x: self._custom_tool(cap, x),
                        description=f"Execute {cap} capability"
                    ))
                return tools
            
            def _custom_tool(self, capability: str, input_data: str) -> str:
                prompt = f"""{self.system_prompt}

Capability: {capability}
Input: {input_data}

Execute this task using your expertise.
"""
                return self.llm.generate(prompt)
            
            def process(self, state: AgentState) -> AgentState:
                logger.info(f"[{self.name}] Processing custom task")
                
                query = state.messages[-1]["content"] if state.messages else ""
                full_prompt = f"""{self.system_prompt}

Context:
{state.rag_context}

Task: {query}
"""
                result = self.llm.generate(full_prompt)
                
                state.messages.append({
                    "role": "assistant",
                    "content": result,
                    "agent": self.name
                })
                state.current_agent = self.name
                return state
        
        agent = DynamicAgent(self.llm)
        self.registry[name] = agent
        logger.info(f"Custom agent created: {name}")
        return agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Retrieve registered custom agent"""
        return self.registry.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered custom agents"""
        return list(self.registry.keys())


# ============================================
# MULTI-AGENT ORCHESTRATOR
# ============================================

class AgentOrchestrator:
    """LangGraph-based multi-agent orchestration with intelligent routing"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.rag = WebScrapingRAG()
        
        # Initialize core agents
        self.agents = {
            "code": CodeArchitect(self.llm),
            "security": SecAnalyst(self.llm),
            "automation": AutoBot(self.llm),
            "admin": AgentSuite(self.llm),
            "creative": CreativeAgent(self.llm),
        }
        
        # Custom agent factory
        self.custom_factory = CustomAgentFactory(self.llm)
        
        # Build orchestration graph
        self.graph = self._build_graph()
        
        logger.info("AgentOrchestrator initialized with all agents")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph orchestration workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("router", self._route_task)
        workflow.add_node("rag", self._rag_retrieval)
        workflow.add_node("code", self.agents["code"].process)
        workflow.add_node("security", self.agents["security"].process)
        workflow.add_node("automation", self.agents["automation"].process)
        workflow.add_node("admin", self.agents["admin"].process)
        workflow.add_node("creative", self.agents["creative"].process)
        workflow.add_node("finalizer", self._finalize)
        
        # Define edges
        workflow.set_entry_point("rag")
        workflow.add_edge("rag", "router")
        
        # Router to agents
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "code": "code",
                "security": "security",
                "automation": "automation",
                "admin": "admin",
                "creative": "creative",
                "custom": "router",  # Handle custom agents
                "end": "finalizer"
            }
        )
        
        # Agents back to router or finalizer
        for agent_name in ["code", "security", "automation", "admin", "creative"]:
            workflow.add_conditional_edges(
                agent_name,
                self._continue_or_end,
                {
                    "continue": "router",
                    "end": "finalizer"
                }
            )
        
        workflow.add_edge("finalizer", END)
        
        return workflow.compile()
    
    def _rag_retrieval(self, state: AgentState) -> AgentState:
        """RAG retrieval node"""
        logger.info("[RAG] Retrieving context")
        
        query = state.messages[-1]["content"] if state.messages else ""
        state.rag_context = self.rag.query(query)
        
        return state
    
    def _route_task(self, state: AgentState) -> AgentState:
        """Intelligent task routing"""
        logger.info("[Router] Routing task")
        
        query = state.messages[-1]["content"] if state.messages else ""
        
        routing_prompt = f"""Analyze this task and determine which agent should handle it:

Task: {query}

Available agents:
- code: Python, JS, Rust, Go, API development
- security: Pen testing, audits, threat modeling
- automation: Workflows, API integrations
- admin: Reports, finance, operations
- creative: Content creation, writing

Respond with ONLY the agent name.
"""
        
        decision = self.llm.generate(routing_prompt).lower().strip()
        
        # Validate decision
        valid_agents = ["code", "security", "automation", "admin", "creative"]
        if decision not in valid_agents:
            decision = "code"  # Default
        
        state.context["next_agent"] = decision
        logger.info(f"[Router] Routed to: {decision}")
        
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Determine next node from router"""
        if state.iteration >= Config.MAX_ITERATIONS:
            return "end"
        
        next_agent = state.context.get("next_agent", "code")
        state.iteration += 1
        
        return next_agent
    
    def _continue_or_end(self, state: AgentState) -> str:
        """Decide if workflow should continue or end"""
        # Check if task is complete
        if state.iteration >= Config.MAX_ITERATIONS:
            return "end"
        
        # Simple heuristic: if last message has conclusive keywords
        last_message = state.messages[-1]["content"].lower()
        conclusive_keywords = ["complete", "done", "finished", "here is", "solution"]
        
        if any(keyword in last_message for keyword in conclusive_keywords):
            return "end"
        
        return "continue"
    
    def _finalize(self, state: AgentState) -> AgentState:
        """Finalize and format output"""
        logger.info("[Finalizer] Creating final output")
        
        # Compile all agent responses
        agent_outputs = [
            f"**{msg['agent']}:**\n{msg['content']}\n"
            for msg in state.messages
            if msg.get("role") == "assistant"
        ]
        
        state.final_output = "\n".join(agent_outputs)
        
        return state
    
    def process(self, user_query: str) -> Dict[str, Any]:
        """Process user query through multi-agent system"""
        logger.info(f"Processing query: {user_query}")
        
        # Initialize state
        initial_state = AgentState(
            messages=[{"role": "user", "content": user_query}],
            iteration=0
        )
        
        # Execute graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "status": "success",
                "output": final_state.final_output,
                "agents_used": [msg.get("agent") for msg in final_state.messages if msg.get("agent")],
                "iterations": final_state.iteration,
                "rag_context_used": bool(final_state.rag_context)
            }
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def create_custom_agent(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        system_prompt: str
    ) -> Dict[str, Any]:
        """Create and register custom agent"""
        try:
            agent = self.custom_factory.create_agent(
                name, description, capabilities, system_prompt
            )
            
            # Add to agents registry
            self.agents[name.lower()] = agent
            
            return {
                "status": "success",
                "message": f"Custom agent '{name}' created successfully",
                "agent_name": name
            }
            
        except Exception as e:
            logger.error(f"Failed to create custom agent: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents"""
        core_agents = {
            name: {
                "type": "core",
                "capability": agent.capability.value,
                "tools": len(agent.tools)
            }
            for name, agent in self.agents.items()
            if agent.capability != AgentCapability.CUSTOM
        }
        
        custom_agents = {
            name: {
                "type": "custom",
                "capability": "custom"
            }
            for name in self.custom_factory.list_agents()
        }
        
        return {
            "core_agents": core_agents,
            "custom_agents": custom_agents,
            "total": len(core_agents) + len(custom_agents)
        }


# ============================================
# METRICS & MONITORING
# ============================================

class SystemMetrics:
    """Production metrics collection"""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.latencies = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_request(self, agent: str, latency: float):
        """Record request metrics"""
        with self.lock:
            self.metrics[f"{agent}_requests"] += 1
            self.latencies[agent].append(latency)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        with self.lock:
            stats = dict(self.metrics)
            
            for agent, latencies in self.latencies.items():
                if latencies:
                    stats[f"{agent}_avg_latency"] = np.mean(latencies)
                    stats[f"{agent}_p95_latency"] = np.percentile(latencies, 95)
            
            return stats


# ============================================
# GLOBAL SYSTEM INSTANCE
# ============================================

_orchestrator_instance = None
_metrics_instance = SystemMetrics()

def get_orchestrator() -> AgentOrchestrator:
    """Singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance

def get_metrics() -> SystemMetrics:
    """Get metrics instance"""
    return _metrics_instance


# ============================================
# EXPORT
# ============================================

__all__ = [
    'AgentOrchestrator',
    'CustomAgentFactory',
    'WebScrapingRAG',
    'get_orchestrator',
    'get_metrics',
    'Config'
]
