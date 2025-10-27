"""
GOOGLE COLAB EXECUTION SCRIPT
Enterprise AI Multi-Agent Orchestration System
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import subprocess
import time
from typing import Optional
import nest_asyncio

# Enable nested event loops for Colab
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Setup Colab environment"""
    logger.info("🚀 Setting up Enterprise AI Multi-Agent System...")
    
    # Create directory structure
    dirs = ['logs', 'vectordb', 'airflow/dags']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("✅ Environment setup complete")


def install_dependencies():
    """Install system dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            check=True
        )
        
        # Install playwright browsers for advanced scraping (optional)
        try:
            subprocess.run(
                ["playwright", "install", "chromium"],
                check=False,
                capture_output=True
            )
        except:
            logger.warning("Playwright browser install skipped")
        
        logger.info("✅ Dependencies installed")
    
    except Exception as e:
        logger.error(f"❌ Dependency installation failed: {e}")
        raise


def validate_config():
    """Validate configuration"""
    logger.info("🔍 Validating configuration...")
    
    required_vars = ['NGROK_AUTH_TOKEN', 'GOOGLE_CSE_URL']
    
    from dotenv import load_dotenv
    load_dotenv()
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing)}")
        logger.info("Please update .env file with required values")
        return False
    
    logger.info("✅ Configuration validated")
    return True


# ============================================================================
# NGROK TUNNEL SETUP
# ============================================================================

def setup_ngrok():
    """Setup ngrok tunnel for API access"""
    logger.info("🌐 Setting up ngrok tunnel...")
    
    try:
        from pyngrok import ngrok, conf
        
        # Set auth token
        auth_token = os.getenv('NGROK_AUTH_TOKEN')
        conf.get_default().auth_token = auth_token
        
        # Kill existing tunnels
        ngrok.kill()
        
        # Create HTTP tunnel
        api_port = int(os.getenv('API_PORT', 8080))
        tunnel = ngrok.connect(api_port, "http")
        
        logger.info(f"✅ Ngrok tunnel active: {tunnel.public_url}")
        logger.info(f"📡 API accessible at: {tunnel.public_url}/docs")
        
        return tunnel
    
    except Exception as e:
        logger.error(f"❌ Ngrok setup failed: {e}")
        return None


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

async def initialize_model():
    """Initialize Qwen3-1.7B model"""
    logger.info("🤖 Initializing Qwen3-1.7B model...")
    
    try:
        from system import LLMEngine
        
        # This will download and load the model
        engine = LLMEngine()
        
        # Test generation
        test_response = await engine.generate("Hello! System check.", max_tokens=50)
        logger.info(f"✅ Model initialized. Test response: {test_response[:100]}...")
        
        return engine
    
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise


# ============================================================================
# FASTAPI SERVER
# ============================================================================

class ServerManager:
    """Manage FastAPI server lifecycle"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.tunnel = None
    
    def start(self):
        """Start FastAPI server"""
        logger.info("🚀 Starting FastAPI server...")
        
        try:
            api_host = os.getenv('API_HOST', '0.0.0.0')
            api_port = int(os.getenv('API_PORT', 8080))
            
            # Start uvicorn server
            self.process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "system:app",
                    "--host", api_host,
                    "--port", str(api_port),
                    "--workers", "1",
                    "--log-level", "info"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            if self.process.poll() is None:
                logger.info(f"✅ FastAPI server running on http://{api_host}:{api_port}")
                
                # Setup ngrok tunnel
                self.tunnel = setup_ngrok()
                
                return True
            else:
                logger.error("❌ FastAPI server failed to start")
                return False
        
        except Exception as e:
            logger.error(f"❌ Server start failed: {e}")
            return False
    
    def stop(self):
        """Stop FastAPI server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("🛑 FastAPI server stopped")
        
        if self.tunnel:
            from pyngrok import ngrok
            ngrok.kill()
            logger.info("🛑 Ngrok tunnel closed")


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

async def run_interactive_demo():
    """Run interactive demo"""
    logger.info("\n" + "="*70)
    logger.info("🎯 ENTERPRISE AI MULTI-AGENT SYSTEM - INTERACTIVE DEMO")
    logger.info("="*70 + "\n")
    
    from system import MultiAgentOrchestrator
    
    orchestrator = MultiAgentOrchestrator()
    
    demo_tasks = [
        {
            'task': 'Create a FastAPI endpoint for user authentication with JWT tokens',
            'task_type': 'code',
            'description': 'Code generation task'
        },
        {
            'task': 'Audit this authentication endpoint for security vulnerabilities',
            'task_type': 'security',
            'description': 'Security analysis task'
        },
        {
            'task': 'Design an Airflow DAG to automate daily data pipeline',
            'task_type': 'automation',
            'description': 'Automation workflow task'
        },
        {
            'task': 'Generate a marketing blog post about AI automation benefits',
            'task_type': 'content',
            'description': 'Content creation task'
        }
    ]
    
    for idx, demo in enumerate(demo_tasks, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"📋 DEMO {idx}/{len(demo_tasks)}: {demo['description']}")
        logger.info(f"{'='*70}")
        logger.info(f"Task: {demo['task']}")
        logger.info(f"Type: {demo['task_type']}")
        logger.info("\n⏳ Processing...\n")
        
        result = await orchestrator.process_task(
            task=demo['task'],
            task_type=demo['task_type'],
            enable_rag=True
        )
        
        logger.info(f"✅ Status: {result['status']}")
        logger.info(f"🤖 Agent: {result['agent']}")
        logger.info(f"📚 RAG Contexts: {result['context_used']}")
        
        if result['output']:
            logger.info(f"\n📤 OUTPUT:\n{'-'*70}")
            logger.info(result['output'][:500] + "..." if len(result['output']) > 500 else result['output'])
            logger.info(f"{'-'*70}\n")
        
        if result['errors']:
            logger.warning(f"⚠️ Errors: {result['errors']}")
        
        await asyncio.sleep(2)
    
    logger.info("\n" + "="*70)
    logger.info("✅ DEMO COMPLETED")
    logger.info("="*70 + "\n")


# ============================================================================
# MONITORING DASHBOARD
# ============================================================================

def display_dashboard_info(tunnel_url: Optional[str] = None):
    """Display dashboard access information"""
    logger.info("\n" + "="*70)
    logger.info("📊 DASHBOARD ACCESS")
    logger.info("="*70)
    
    if tunnel_url:
        logger.info(f"\n🌐 Public API URL: {tunnel_url}")
        logger.info(f"📖 API Documentation: {tunnel_url}/docs")
        logger.info(f"📈 Metrics: {tunnel_url}/metrics")
        logger.info(f"💚 Health Check: {tunnel_url}/api/v1/health")
        logger.info(f"🤖 Agents List: {tunnel_url}/api/v1/agents")
    
    logger.info(f"\n📂 Local Dashboard: file://{Path('dashboard.html').absolute()}")
    logger.info("\n" + "="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution flow"""
    
    try:
        # Setup
        setup_environment()
        install_dependencies()
        
        if not validate_config():
            logger.error("Configuration validation failed. Please check .env file.")
            return
        
        # Initialize model
        await initialize_model()
        
        # Start server
        server = ServerManager()
        if not server.start():
            logger.error("Failed to start server")
            return
        
        # Display access info
        display_dashboard_info(server.tunnel.public_url if server.tunnel else None)
        
        # Run interactive demo
        await run_interactive_demo()
        
        # Keep server running
        logger.info("\n🔄 Server running. Press Ctrl+C to stop.\n")
        logger.info("💡 Use the API endpoints to interact with agents:")
        logger.info("   - POST /api/v1/task - Execute agent task")
        logger.info("   - GET /api/v1/health - Health check")
        logger.info("   - GET /api/v1/agents - List agents\n")
        
        # Keep alive
        try:
            while True:
                await asyncio.sleep(60)
                logger.info("💓 System heartbeat - All systems operational")
        except KeyboardInterrupt:
            logger.info("\n⚠️ Shutdown signal received")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if 'server' in locals():
            server.stop()
        logger.info("👋 Shutdown complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Entry point for Colab execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ENTERPRISE AI MULTI-AGENT ORCHESTRATION SYSTEM              ║
    ║  Production-Grade Agentic Platform                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run main async function
    asyncio.run(main())
