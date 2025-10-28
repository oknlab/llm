import os
import subprocess
from typing import List, Annotated, Literal
from typing_extensions import TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders.base import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORGMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import asyncio

# --- 1. CONFIGURATION & INITIALIZATION ---
load_dotenv()

# Load API configuration from environment variables
API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
RAG_CSE_URL = "https://cse.google.com/cse?cx=014662525286492529401%3A2upbuo2qpni"

# --- 2. RAG TOOL DEFINITION ---
# This tool performs live web searches using the specified Google Custom Search Engine.
@tool("web_search_rag")
def web_search_rag(query: str) -> str:
    """
    Performs a real-time web search using a custom search engine to find relevant code,
    security information, or other specified data. Returns a compiled list of findings.
    """
    print(f"--- INFO: Executing RAG Tool for query: '{query}' ---")
    try:
        response = requests.get(RAG_CSE_URL, params={'q': query}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results (this selector is specific to Google CSE)
        results = soup.find_all('div', class_='gs-webResult gs-result')
        if not results:
            return "No results found from the web search."

        content = []
        for result in results[:5]: # Limit to top 5 results
            title_elem = result.find('a', class_='gs-title')
            snippet_elem = result.find('div', class_='gs-bidi-start-align gs-snippet')
            if title_elem and snippet_elem:
                title = title_elem.text.strip()
                link = title_elem['href']
                snippet = snippet_elem.text.strip()
                content.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
        
        return "\n\n---\n\n".join(content) if content else "Web search executed, but no parsable content found."

    except requests.RequestException as e:
        return f"Error during web search: {e}"

# --- 3. AGENTIC GRAPH STATE & ORCHESTRATION ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sender: str

# Helper function to create a generic agent node. This is our agent factory.
def create_agent_node(llm: ChatOpenAI, system_prompt: str, agent_name: str, tools: List):
    """Factory to create a LangGraph agent node."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    agent_runnable = prompt | llm.bind_tools(tools)

    def agent_node(state: AgentState):
        print(f"--- INFO: Executing Agent Node: {agent_name} ---")
        response = agent_runnable.invoke(state)
        return {"messages": [response], "sender": agent_name}
    
    return agent_node

# Router function to decide the next step in the graph
def router(state: AgentState) -> Literal["CodeArchitect", "SecAnalyst", "CreativeAgent", "AgentSuite", "__end__", "tools"]:
    """Routes the workflow to the appropriate agent or ends the process."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # If the LLM decides to use a tool, route to the tool node
        return "tools"

    # Route based on content and sender
    content = last_message.content.lower()
    if "code" in content or "python" in content or "api" in content or "rust" in content or "go" in content:
        return "CodeArchitect"
    if "security" in content or "audit" in content or "threat" in content:
        return "SecAnalyst"
    if "report" in content or "automate ops" in content:
        return "AgentSuite"
    if "write" in content or "create content" in content:
        return "CreativeAgent"
    
    # Default exit condition
    return "__end__"

# --- 4. CORE APPLICATION SETUP (FastAPI + LangGraph) ---

def get_app():
    """Constructs and returns the compiled LangGraph app and FastAPI instance."""
    
    # LLM and Tools setup
    llm = ChatOpenAI(model=MODEL_ID, openai_api_base=API_BASE, openai_api_key=API_KEY, temperature=0.1)
    tools = [web_search_rag]
    tool_node = ToolNode(tools)

    # Agent Definitions using the factory
    code_architect_node = create_agent_node(
        llm, "You are the CodeArchitect. Your role is to design and write clean, modular, and optimized code. Use the web_search_rag tool to find libraries or best practices. Provide full, complete code files. Do not use placeholders.", "CodeArchitect", tools)
    
    sec_analyst_node = create_agent_node(
        llm, "You are the SecAnalyst. Your role is to audit code for security vulnerabilities, model threats, and suggest robust security practices.", "SecAnalyst", [])
    
    agent_suite_node = create_agent_node(
        llm, "You are the AgentSuite admin. Your role is to generate reports, and automate operational workflows.", "AgentSuite", [])
    
    creative_agent_node = create_agent_node(
        llm, "You are the CreativeAgent. You generate high-quality text, audio, or visual content outlines based on user requests.", "CreativeAgent", [])

    # Graph construction
    graph = StateGraph(AgentState)
    graph.add_node("CodeArchitect", code_architect_node)
    graph.add_node("SecAnalyst", sec_analyst_node)
    graph.add_node("AgentSuite", agent_suite_node)
    graph.add_node("CreativeAgent", creative_agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("CodeArchitect") # Default entry for technical tasks

    graph.add_conditional_edges(
        "CodeArchitect",
        router,
        {"CodeArchitect": "CodeArchitect", "SecAnalyst": "SecAnalyst", "AgentSuite": "AgentSuite", "CreativeAgent": "CreativeAgent", "__end__": END, "tools": "tools"}
    )
    # Add edges from other agents and tools back to the router
    graph.add_edge("SecAnalyst", END) # Simplified flow for this example
    graph.add_edge("AgentSuite", END)
    graph.add_edge("CreativeAgent", END)
    graph.add_edge("tools", "CodeArchitect") # After tool use, return to the main agent

    # Compile the graph into a runnable app
    langgraph_app = graph.compile()
    
    # --- FastAPI App Definition ---
    fastapi_app = FastAPI(
        title="AI Core Agentic Platform",
        description="An enterprise-grade, multi-agent AI system.",
        version="1.0.0"
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class InvokeRequest(BaseModel):
        prompt: str

    @fastapi_app.post("/invoke")
    async def invoke_agent(request: InvokeRequest):
        """
        Endpoint to invoke the agent graph with a user prompt.
        Streams the final response back to the client.
        """
        inputs = {"messages": [HumanMessage(content=request.prompt)], "sender": "user"}
        
        async def stream_events():
            async for event in langgraph_app.astream(inputs):
                if "messages" in event.get("CodeArchitect", {}):
                    last_message = event["CodeArchitect"]["messages"][-1]
                    if isinstance(last_message, AIMessage) and last_message.content:
                        yield f"data: {last_message.content}\n\n"
                        await asyncio.sleep(0.01) # Yield control
            yield "data: [END_OF_STREAM]\n\n"

        return EventSourceResponse(stream_events())

    @fastapi_app.get("/status")
    def get_system_status():
        """
        Provides system status without using `psutil`.
        Uses `os` and `subprocess` for basic metrics.
        """
        try:
            # CPU Load (Unix-specific)
            load1, load5, load15 = os.getloadavg()
            cpu_load = {"1min": load1, "5min": load5, "15min": load15}

            # Memory Usage (Unix-specific, requires `free`)
            mem_info_raw = subprocess.check_output("free -m", shell=True).decode()
            lines = mem_info_raw.split('\n')
            mem_line = lines[1].split()
            total_mem, used_mem = int(mem_line[1]), int(mem_line[2])
            mem_usage = {"total_mb": total_mem, "used_mb": used_mem, "percent": round((used_mem / total_mem) * 100, 2)}
            
        except Exception:
            # Fallback for non-Unix or restricted environments
            cpu_load = "N/A"
            mem_usage = "N/A"

        return {
            "status": "ok",
            "model_id": MODEL_ID,
            "api_base": API_BASE,
            "cpu_load": cpu_load,
            "memory_usage": mem_usage,
            "active_agents": list(graph.nodes.keys())
        }

    return langgraph_app, fastapi_app

# Main execution guarded to prevent running on import
if __name__ == "__main__":
    import uvicorn
    _, app = get_app()
    print("--- System Core Ready. Starting FastAPI server on http://127.0.0.1:8001 ---")
    uvicorn.run(app, host="127.0.0.1", port=8001)
