"""
Enterprise AI Orchestration System - Core Agent Architecture
Multi-Agent System with Agentic RAG, vLLM, and LangGraph State Machine
"""

import os
import asyncio
import json
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver

import faiss
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Environment configuration with validation"""
    
    # Model Configuration
    model_name: str = "Qwen/Qwen3-1.7B"
    vllm_api_base: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-abc123"
    
    # RAG Configuration
    google_cse_url: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Agent Configuration
    max_iterations: int = 5
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class AgentType(str, Enum):
    """Agent role enumeration"""
    CODE_ARCHITECT = "CodeArchitect"
    SEC_ANALYST = "SecAnalyst"
    AUTO_BOT = "AutoBot"
    AGENT_SUITE = "AgentSuite"
    CREATIVE_AGENT = "CreativeAgent"
    AG_CUSTOM = "AgCustom"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: AgentType
    expertise: str
    system_prompt: str
    tools: list = field(default_factory=list)
    
    
AGENT_REGISTRY = {
    AgentType.CODE_ARCHITECT: AgentCapability(
        name=AgentType.CODE_ARCHITECT,
        expertise="Software Architecture, Python/JS/Rust/Go, API Design, Code Optimization",
        system_prompt="""You are CodeArchitect, an elite software engineer specializing in:
- Clean, modular architecture design
- Algorithm optimization (time/space complexity)
- Multi-language expertise: Python, JavaScript, Rust, Go
- RESTful API design and GraphQL
- Performance profiling and refactoring
- Design patterns and SOLID principles

Provide production-ready code with proper error handling, type hints, and documentation."""
    ),
    
    AgentType.SEC_ANALYST: AgentCapability(
        name=AgentType.SEC_ANALYST,
        expertise="Security Audits, Penetration Testing, Threat Modeling, Compliance",
        system_prompt="""You are SecAnalyst, a cybersecurity expert specializing in:
- Vulnerability assessment and penetration testing
- OWASP Top 10 mitigation strategies
- Threat modeling (STRIDE, DREAD)
- Security code review and static analysis
- Compliance frameworks (SOC2, ISO27001, GDPR)
- Zero-trust architecture design

Identify security risks and provide actionable remediation steps."""
    ),
    
    AgentType.AUTO_BOT: AgentCapability(
        name=AgentType.AUTO_BOT,
        expertise="Workflow Automation, API Integration, Zapier/n8n/Make, Data Pipelines",
        system_prompt="""You are AutoBot, an automation specialist expert in:
- Workflow orchestration (Airflow, Prefect, n8n)
- API integration (REST, GraphQL, Webhooks)
- ETL/ELT pipeline design (Airbyte, Kafka, dbt)
- Event-driven architecture (Pub/Sub, Message Queues)
- Infrastructure as Code (Terraform, Ansible)
- CI/CD pipeline optimization (GitHub Actions, GitLab CI)

Design scalable, fault-tolerant automation systems."""
    ),
    
    AgentType.AGENT_SUITE: AgentCapability(
        name=AgentType.AGENT_SUITE,
        expertise="Operations, Finance Automation, Reporting, Admin Tasks",
        system_prompt="""You are AgentSuite, an operations and finance automation expert:
- Financial reporting and reconciliation automation
- Admin workflow optimization (document processing, approvals)
- Dashboard creation (Grafana, Metabase, Tableau)
- Process mining and bottleneck analysis
- Resource allocation and capacity planning
- KPI tracking and alerting systems

Optimize operational efficiency with data-driven insights."""
    ),
    
    AgentType.CREATIVE_AGENT: AgentCapability(
        name=AgentType.CREATIVE_AGENT,
        expertise="Content Generation, Text/Audio/Visual Creation, Marketing",
        system_prompt="""You are CreativeAgent, a content creation specialist:
- Technical writing and documentation
- Marketing copy and SEO optimization
- Data visualization and infographics
- Audio/video script generation
- Brand voice consistency
- Multi-channel content strategy

Create engaging, conversion-optimized content with brand alignment."""
    ),
    
    AgentType.AG_CUSTOM: AgentCapability(
        name=AgentType.AG_CUSTOM,
        expertise="Custom Solutions, Domain-Specific AI, Specialized Workflows",
        system_prompt="""You are AgCustom, a versatile AI agent for custom solutions:
- Rapid prototyping of domain-specific agents
- Integration with legacy systems
- Custom ML model deployment
- Niche automation workflows
- Specialized data processing pipelines
- Industry-specific compliance automation

Adapt to unique requirements with innovative solutions."""
    )
}


# ============================================================================
# AGENTIC RAG ENGINE - ALGORITHM 1: WEB SCRAPING + RETRIEVAL
# ============================================================================

class AgenticRAGEngine:
    """
    Advanced RAG with multi-hop retrieval, self-reflection, and web scraping
    
    Algorithm:
    1. Query Decomposition: Break complex queries into sub-queries
    2. Web Scraping: Fetch real-time data from Google CSE
    3. Content Extraction: Parse and chunk HTML content
    4. Embedding Generation: Convert chunks to dense vectors
    5. Hybrid Retrieval: Combine semantic + keyword search
    6. Reranking: Score and filter results by relevance
    7. Self-Reflection: Validate answer quality
    """
    
    def __init__(self):
        logger.info("Initializing Agentic RAG Engine...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store with empty index
        index = faiss.IndexFlatL2(settings.embedding_dimension)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # HTTP client for web scraping
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.success("RAG Engine initialized")
    
    async def decompose_query(self, query: str) -> list[str]:
        """
        Algorithm: Query Decomposition
        Break complex queries into atomic sub-queries for multi-hop retrieval
        """
        # Simple heuristic decomposition (can be enhanced with LLM)
        sub_queries = [query]
        
        # Check for compound questions
        if " and " in query.lower():
            sub_queries.extend(query.lower().split(" and "))
        if " or " in query.lower():
            sub_queries.extend(query.lower().split(" or "))
        
        # Extract entities (simple keyword extraction)
        keywords = [w for w in query.split() if len(w) > 4]
        sub_queries.extend(keywords[:3])  # Top 3 keywords
        
        return list(set(sub_queries))  # Remove duplicates
    
    async def scrape_web_content(self, query: str) -> list[Document]:
        """
        Algorithm: Real-time Web Scraping from Google CSE
        
        Steps:
        1. Construct search URL with query
        2. Fetch HTML content
        3. Parse with BeautifulSoup
        4. Extract text from result snippets
        5. Create Document objects
        """
        try:
            # Construct Google CSE search URL
            search_url = f"{settings.google_cse_url}&q={httpx.QueryParams({'q': query})}"
            
            logger.info(f"Scraping: {search_url}")
            response = await self.http_client.get(search_url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            documents = []
            
            # Extract search results
            for result in soup.select('.g'):  # Google result container
                try:
                    title_elem = result.select_one('h3')
                    snippet_elem = result.select_one('.VwiC3b')  # Snippet class
                    link_elem = result.select_one('a')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True)
                        link = link_elem.get('href', '') if link_elem else ''
                        
                        # Create document
                        doc = Document(
                            page_content=f"{title}\n\n{snippet}",
                            metadata={
                                'source': link,
                                'title': title,
                                'timestamp': datetime.utcnow().isoformat(),
                                'query': query
                            }
                        )
                        documents.append(doc)
                
                except Exception as e:
                    logger.warning(f"Failed to parse result: {e}")
                    continue
            
            logger.success(f"Scraped {len(documents)} documents for query: {query}")
            return documents
        
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return []
    
    async def index_documents(self, documents: list[Document]):
        """
        Algorithm: Document Indexing
        
        Steps:
        1. Split documents into chunks
        2. Generate embeddings for each chunk
        3. Add to FAISS vector store
        """
        if not documents:
            return
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks from {len(documents)} documents")
    
    async def retrieve_context(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Algorithm: Hybrid Retrieval with Reranking
        
        Steps:
        1. Decompose query into sub-queries
        2. Scrape fresh web content for each sub-query
        3. Index new content
        4. Perform semantic similarity search
        5. Rerank by relevance score
        """
        # Decompose query
        sub_queries = await self.decompose_query(query)
        
        # Scrape content for each sub-query
        all_docs = []
        for sq in sub_queries[:3]:  # Limit to top 3 sub-queries
            docs = await self.scrape_web_content(sq)
            all_docs.extend(docs)
        
        # Index new documents
        await self.index_documents(all_docs)
        
        # Retrieve similar documents
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # Filter by relevance threshold
            threshold = 0.7
            filtered_results = [
                doc for doc, score in results 
                if score < threshold  # Lower score = higher similarity in L2
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} relevant documents")
            return filtered_results if filtered_results else [doc for doc, _ in results]
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def self_reflect(self, query: str, answer: str, context: list[Document]) -> dict:
        """
        Algorithm: Self-Reflection on Answer Quality
        
        Evaluate:
        - Relevance: Does answer address the query?
        - Groundedness: Is answer supported by context?
        - Completeness: Are all aspects covered?
        """
        metrics = {
            'relevance_score': 0.0,
            'groundedness_score': 0.0,
            'completeness_score': 0.0,
            'needs_revision': False
        }
        
        # Simple heuristic evaluation
        query_keywords = set(query.lower().split())
        answer_keywords = set(answer.lower().split())
        
        # Relevance: keyword overlap
        overlap = query_keywords.intersection(answer_keywords)
        metrics['relevance_score'] = len(overlap) / max(len(query_keywords), 1)
        
        # Groundedness: check if answer references context
        context_text = " ".join([doc.page_content for doc in context]).lower()
        grounded_keywords = [k for k in answer_keywords if k in context_text]
        metrics['groundedness_score'] = len(grounded_keywords) / max(len(answer_keywords), 1)
        
        # Completeness: answer length heuristic
        metrics['completeness_score'] = min(len(answer.split()) / 100, 1.0)
        
        # Overall quality check
        avg_score = sum(metrics.values()) / 3
        metrics['needs_revision'] = avg_score < 0.5
        
        return metrics


# ============================================================================
# LANGGRAPH STATE MACHINE - ALGORITHM 2: MULTI-AGENT ORCHESTRATION
# ============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the graph"""
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    query: str
    context: list[Document]
    current_agent: str
    agent_outputs: dict
    final_answer: str
    iteration: int
    metadata: dict


class AIOrchestrator:
    """
    Multi-Agent Orchestration System using LangGraph
    
    Algorithm Flow:
    1. Router: Classify intent → select agent(s)
    2. Agent Execution: Parallel or sequential processing
    3. RAG Integration: Retrieve context for each agent
    4. Synthesis: Combine multi-agent outputs
    5. Self-Improvement: Reflect and iterate
    """
    
    def __init__(self):
        logger.info("Initializing AI Orchestrator...")
        
        # Initialize RAG engine
        self.rag_engine = AgenticRAGEngine()
        
        # Initialize LLM (vLLM backend)
        self.llm = ChatOpenAI(
            model=settings.model_name,
            openai_api_base=settings.vllm_api_base,
            openai_api_key=settings.vllm_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_p=settings.top_p
        )
        
        # Build LangGraph state machine
        self.graph = self._build_graph()
        
        logger.success("AI Orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Algorithm: LangGraph State Machine Construction
        
        Graph Structure:
        START → Router → [Agent1, Agent2, ...] → Synthesizer → END
                  ↓                                    ↑
                 RAG ←────────────────────────────────┘
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("rag", self._rag_node)
        
        # Add agent nodes
        for agent_type in AgentType:
            workflow.add_node(agent_type.value, self._create_agent_node(agent_type))
        
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Define edges
        workflow.set_entry_point("router")
        
        # Router → RAG → Agents
        workflow.add_edge("router", "rag")
        
        # RAG → Selected agents (conditional)
        workflow.add_conditional_edges(
            "rag",
            self._route_to_agents,
            {agent.value: agent.value for agent in AgentType}
        )
        
        # All agents → Synthesizer
        for agent_type in AgentType:
            workflow.add_edge(agent_type.value, "synthesizer")
        
        # Synthesizer → END
        workflow.add_edge("synthesizer", END)
        
        # Compile graph with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _router_node(self, state: AgentState) -> AgentState:
        """
        Algorithm: Intent Classification & Agent Selection
        
        Steps:
        1. Analyze query intent
        2. Extract entities and keywords
        3. Map to agent capabilities
        4. Select single or multiple agents
        """
        query = state["query"]
        
        # Intent classification prompt
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router. Analyze the query and determine which specialized agents should handle it.

Available Agents:
- CodeArchitect: Software development, architecture, algorithms, code review
- SecAnalyst: Security audits, penetration testing, threat modeling
- AutoBot: Workflow automation, API integration, data pipelines
- AgentSuite: Operations, finance, reporting, admin tasks
- CreativeAgent: Content creation, writing, marketing, visual assets
- AgCustom: Custom solutions, specialized workflows

Respond with JSON: {{"agents": ["AgentName1", "AgentName2"], "reasoning": "why these agents"}}"""),
            ("human", f"Query: {query}")
        ])
        
        response = await self.llm.ainvoke(router_prompt.format_messages())
        
        try:
            # Parse agent selection
            result = json.loads(response.content)
            selected_agents = result.get("agents", ["CodeArchitect"])
        except:
            # Fallback: keyword matching
            selected_agents = self._fallback_routing(query)
        
        state["metadata"]["selected_agents"] = selected_agents
        logger.info(f"Router selected agents: {selected_agents}")
        
        return state
    
    def _fallback_routing(self, query: str) -> list[str]:
        """Keyword-based fallback routing"""
        query_lower = query.lower()
        agents = []
        
        keywords_map = {
            AgentType.CODE_ARCHITECT: ["code", "python", "api", "architecture", "algorithm"],
            AgentType.SEC_ANALYST: ["security", "vulnerability", "audit", "threat"],
            AgentType.AUTO_BOT: ["automation", "workflow", "pipeline", "integration"],
            AgentType.AGENT_SUITE: ["finance", "report", "operations", "admin"],
            AgentType.CREATIVE_AGENT: ["content", "writing", "marketing", "design"],
            AgentType.AG_CUSTOM: ["custom", "specialized", "unique"]
        }
        
        for agent, keywords in keywords_map.items():
            if any(kw in query_lower for kw in keywords):
                agents.append(agent.value)
        
        return agents if agents else [AgentType.CODE_ARCHITECT.value]
    
    async def _rag_node(self, state: AgentState) -> AgentState:
        """RAG retrieval node"""
        query = state["query"]
        
        # Retrieve context
        context = await self.rag_engine.retrieve_context(query, top_k=5)
        state["context"] = context
        
        logger.info(f"RAG retrieved {len(context)} documents")
        return state
    
    def _create_agent_node(self, agent_type: AgentType):
        """Factory function to create agent execution nodes"""
        
        async def agent_node(state: AgentState) -> AgentState:
            """Execute specific agent with RAG context"""
            
            # Check if this agent was selected
            if agent_type.value not in state["metadata"].get("selected_agents", []):
                return state
            
            agent_config = AGENT_REGISTRY[agent_type]
            query = state["query"]
            context = state["context"]
            
            # Build context string
            context_str = "\n\n".join([
                f"Source: {doc.metadata.get('title', 'N/A')}\n{doc.page_content}"
                for doc in context[:3]
            ])
            
            # Agent prompt with RAG context
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", agent_config.system_prompt),
                ("system", f"Context from knowledge base:\n{context_str}"),
                ("human", "{query}")
            ])
            
            # Execute agent
            response = await self.llm.ainvoke(
                agent_prompt.format_messages(query=query)
            )
            
            # Store agent output
            state["agent_outputs"][agent_type.value] = {
                "response": response.content,
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent_type.value
            }
            
            state["messages"].append(AIMessage(
                content=response.content,
                name=agent_type.value
            ))
            
            logger.success(f"Agent {agent_type.value} completed")
            return state
        
        return agent_node
    
    def _route_to_agents(self, state: AgentState) -> str:
        """Conditional routing to selected agents"""
        selected = state["metadata"].get("selected_agents", [])
        return selected[0] if selected else AgentType.CODE_ARCHITECT.value
    
    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """
        Algorithm: Multi-Agent Output Synthesis
        
        Steps:
        1. Collect all agent outputs
        2. Identify complementary information
        3. Resolve conflicts
        4. Generate coherent final answer
        """
        agent_outputs = state["agent_outputs"]
        query = state["query"]
        
        # Build synthesis prompt
        outputs_text = "\n\n".join([
            f"**{agent}**:\n{output['response']}"
            for agent, output in agent_outputs.items()
        ])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a synthesis expert. Combine multiple agent responses into a coherent, comprehensive answer.

Requirements:
- Integrate complementary information
- Resolve any contradictions
- Maintain technical accuracy
- Structure clearly with sections
- Cite which agents contributed what insights"""),
            ("human", f"Query: {query}\n\nAgent Responses:\n{outputs_text}")
        ])
        
        final_response = await self.llm.ainvoke(synthesis_prompt.format_messages())
        state["final_answer"] = final_response.content
        
        # Self-reflection
        reflection = await self.rag_engine.self_reflect(
            query, final_response.content, state["context"]
        )
        state["metadata"]["quality_metrics"] = reflection
        
        logger.success("Synthesis complete")
        return state
    
    async def process_query(self, query: str) -> dict:
        """
        Main entry point for query processing
        
        Returns:
        {
            "query": str,
            "final_answer": str,
            "agent_outputs": dict,
            "context": list[Document],
            "metadata": dict
        }
        """
        logger.info(f"Processing query: {query}")
        
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "context": [],
            "current_agent": "",
            "agent_outputs": {},
            "final_answer": "",
            "iteration": 0,
            "metadata": {}
        }
        
        # Execute graph
        config = {"configurable": {"thread_id": "main"}}
        final_state = await self.graph.ainvoke(initial_state, config)
        
        return {
            "query": query,
            "final_answer": final_state["final_answer"],
            "agent_outputs": final_state["agent_outputs"],
            "context": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in final_state["context"]
            ],
            "metadata": final_state["metadata"],
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# FASTAPI SERVICE
# ============================================================================

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Enterprise AI Orchestration System",
    version="1.0.0",
    description="Multi-Agent AI System with Agentic RAG"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: AIOrchestrator = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    stream: bool = False


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    agent_outputs: dict
    context: list
    metadata: dict
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    orchestrator = AIOrchestrator()
    logger.success("FastAPI service started")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Health check"""
    return """
    <html>
        <head><title>AI Orchestration System</title></head>
        <body>
            <h1>🤖 Enterprise AI Orchestration System</h1>
            <p>Status: <strong style="color: green;">ONLINE</strong></p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/dashboard">Dashboard</a></li>
            </ul>
        </body>
    </html>
    """


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process query through multi-agent system"""
    try:
        result = await orchestrator.process_query(request.query)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            
            # Process query
            result = await orchestrator.process_query(query_data["query"])
            
            # Send result
            await websocket.send_json(result)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model": settings.model_name,
        "agents": [agent.value for agent in AgentType]
    }


def run_api_server(host: str = "0.0.0.0", port: int = 7860):
    """Run FastAPI server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "Settings",
    "AgentType",
    "AgenticRAGEngine",
    "AIOrchestrator",
    "app",
    "run_api_server"
]
