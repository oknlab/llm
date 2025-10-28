"""
FIXED: Google Colab Orchestration Script
Proper async execution with health checks and retry logic
"""

import os
import sys
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional
import socket

from loguru import logger
from pyngrok import ngrok, conf
from dotenv import load_dotenv

# Load environment
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class ColabConfig:
    """Colab deployment configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    ENV_FILE = BASE_DIR / ".env"
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    VLLM_HOST = "127.0.0.1"
    VLLM_PORT = 8000
    
    # API Configuration
    API_HOST = "127.0.0.1"
    API_PORT = 7860
    
    # Dashboard
    DASHBOARD_PORT = 8080
    
    # NGROK Configuration
    NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    NGROK_REGION = os.getenv("NGROK_REGION", "us")
    
    # vLLM Performance
    TENSOR_PARALLEL = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    
    # Timeouts
    VLLM_STARTUP_TIMEOUT = 120
    API_STARTUP_TIMEOUT = 30
    HEALTH_CHECK_INTERVAL = 5
    MAX_RETRIES = 10


config = ColabConfig()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 60) -> bool:
    """Wait for port to be open"""
    logger.info(f"Waiting for {host}:{port} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                logger.success(f"Port {port} is ready!")
                return True
        except:
            pass
        time.sleep(2)
    
    logger.error(f"Timeout waiting for port {port}")
    return False


async def check_http_endpoint(url: str, timeout: int = 60) -> bool:
    """Check if HTTP endpoint is responding"""
    import httpx
    
    logger.info(f"Checking endpoint: {url}")
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.success(f"Endpoint {url} is healthy!")
                    return True
            except Exception as e:
                logger.debug(f"Endpoint not ready: {e}")
            
            await asyncio.sleep(3)
    
    logger.error(f"Endpoint {url} failed health check")
    return False


# ============================================================================
# VLLM SERVER MANAGER
# ============================================================================

class VLLMServerManager:
    """Manage vLLM server lifecycle"""
    
    def __init__(self):
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
    
    @staticmethod
    def check_gpu():
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.success(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                return True
            else:
                logger.warning("No GPU - using CPU (very slow)")
                return False
        except ImportError:
            logger.error("PyTorch not installed")
            return False
    
    def start_server(self):
        """Start vLLM server in background thread"""
        
        logger.info("=" * 70)
        logger.info("Starting vLLM Inference Server...")
        logger.info("=" * 70)
        
        # Check GPU
        has_gpu = self.check_gpu()
        
        def run_vllm():
            """Run vLLM in thread"""
            try:
                from vllm.entrypoints.openai.api_server import run_server
                import uvicorn
                
                # vLLM configuration
                os.environ["VLLM_HOST"] = config.VLLM_HOST
                os.environ["VLLM_PORT"] = str(config.VLLM_PORT)
                
                # Create vLLM app
                from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
                from vllm.entrypoints.openai.api_server import app
                
                # Update environment for vLLM
                os.environ.update({
                    "VLLM_MODEL": config.MODEL_NAME,
                    "VLLM_TENSOR_PARALLEL_SIZE": str(config.TENSOR_PARALLEL),
                    "VLLM_GPU_MEMORY_UTILIZATION": str(config.GPU_MEMORY),
                    "VLLM_MAX_MODEL_LEN": str(config.MAX_MODEL_LEN)
                })
                
                logger.info(f"Starting vLLM on {config.VLLM_HOST}:{config.VLLM_PORT}")
                
                # Run uvicorn server
                uvicorn.run(
                    "vllm.entrypoints.openai.api_server:app",
                    host=config.VLLM_HOST,
                    port=config.VLLM_PORT,
                    log_level="info",
                    access_log=False
                )
                
            except Exception as e:
                logger.error(f"vLLM server error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start in daemon thread
        self.server_thread = threading.Thread(target=run_vllm, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        # Wait for server to be ready
        logger.info("Waiting for vLLM server to initialize...")
        time.sleep(10)  # Initial wait
        
        if wait_for_port(config.VLLM_PORT, config.VLLM_HOST, config.VLLM_STARTUP_TIMEOUT):
            logger.success("✅ vLLM server is running!")
            return True
        else:
            logger.error("❌ vLLM server failed to start")
            return False


# ============================================================================
# FASTAPI SERVER MANAGER
# ============================================================================

class FastAPIManager:
    """Manage FastAPI service"""
    
    def __init__(self):
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
    
    def start_server(self):
        """Start FastAPI server in background thread"""
        
        logger.info("=" * 70)
        logger.info("Starting FastAPI Service...")
        logger.info("=" * 70)
        
        def run_api():
            """Run FastAPI in thread"""
            try:
                import uvicorn
                from system import app
                
                logger.info(f"Starting FastAPI on {config.API_HOST}:{config.API_PORT}")
                
                uvicorn.run(
                    app,
                    host=config.API_HOST,
                    port=config.API_PORT,
                    log_level="info",
                    access_log=True
                )
                
            except Exception as e:
                logger.error(f"FastAPI error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start in daemon thread
        self.server_thread = threading.Thread(target=run_api, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        # Wait for server
        logger.info("Waiting for FastAPI to start...")
        time.sleep(5)
        
        if wait_for_port(config.API_PORT, config.API_HOST, config.API_STARTUP_TIMEOUT):
            logger.success("✅ FastAPI server is running!")
            return True
        else:
            logger.error("❌ FastAPI server failed to start")
            return False


# ============================================================================
# DASHBOARD MANAGER
# ============================================================================

class DashboardManager:
    """Manage HTML dashboard"""
    
    def __init__(self):
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
    
    def start_server(self):
        """Start simple HTTP server for dashboard"""
        
        logger.info("=" * 70)
        logger.info("Starting Dashboard Server...")
        logger.info("=" * 70)
        
        def run_dashboard():
            """Run HTTP server in thread"""
            try:
                import http.server
                import socketserver
                
                os.chdir(config.BASE_DIR)
                
                Handler = http.server.SimpleHTTPRequestHandler
                
                with socketserver.TCPServer((config.API_HOST, config.DASHBOARD_PORT), Handler) as httpd:
                    logger.info(f"Dashboard serving on {config.API_HOST}:{config.DASHBOARD_PORT}")
                    httpd.serve_forever()
                    
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start in daemon thread
        self.server_thread = threading.Thread(target=run_dashboard, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        # Wait for server
        time.sleep(3)
        
        if wait_for_port(config.DASHBOARD_PORT, config.API_HOST, 20):
            logger.success("✅ Dashboard server is running!")
            return True
        else:
            logger.error("❌ Dashboard server failed to start")
            return False


# ============================================================================
# NGROK MANAGER
# ============================================================================

class NGROKManager:
    """Manage NGROK tunnels"""
    
    def __init__(self):
        self.tunnels: dict[str, str] = {}
    
    def setup_auth(self):
        """Configure NGROK authentication"""
        if not config.NGROK_TOKEN:
            logger.error("NGROK_AUTH_TOKEN not set in .env file!")
            logger.info("Get token from: https://dashboard.ngrok.com/get-started/your-authtoken")
            return False
        
        try:
            conf.get_default().auth_token = config.NGROK_TOKEN
            conf.get_default().region = config.NGROK_REGION
            logger.success("✅ NGROK authenticated")
            return True
        except Exception as e:
            logger.error(f"NGROK auth failed: {e}")
            return False
    
    def create_tunnel(self, port: int, name: str) -> Optional[str]:
        """Create NGROK tunnel with retry logic"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Creating NGROK tunnel for port {port} (attempt {attempt + 1}/{max_retries})")
                
                tunnel = ngrok.connect(
                    port,
                    bind_tls=True
                )
                
                public_url = tunnel.public_url
                self.tunnels[name] = public_url
                
                logger.success(f"✅ {name.upper()} tunnel: {public_url}")
                return public_url
                
            except Exception as e:
                logger.warning(f"Tunnel creation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    logger.error(f"Failed to create tunnel for {name}")
                    return None
        
        return None
    
    def cleanup(self):
        """Close all tunnels"""
        logger.info("Closing NGROK tunnels...")
        ngrok.kill()


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class SystemOrchestrator:
    """Main system orchestration"""
    
    def __init__(self):
        self.vllm_manager = VLLMServerManager()
        self.api_manager = FastAPIManager()
        self.dashboard_manager = DashboardManager()
        self.ngrok_manager = NGROKManager()
    
    async def health_check_loop(self):
        """Continuous health monitoring"""
        import httpx
        
        while True:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Check FastAPI
                    try:
                        response = await client.get(f"http://{config.API_HOST}:{config.API_PORT}/health")
                        if response.status_code != 200:
                            logger.warning("FastAPI health check failed")
                    except:
                        logger.error("❌ FastAPI is down!")
                    
                    # Check vLLM
                    try:
                        response = await client.get(f"http://{config.VLLM_HOST}:{config.VLLM_PORT}/health")
                        if response.status_code != 200:
                            logger.warning("vLLM health check failed")
                    except:
                        logger.error("❌ vLLM is down!")
                
            except Exception as e:
                logger.debug(f"Health check error: {e}")
            
            await asyncio.sleep(30)
    
    async def start(self):
        """Start all services"""
        
        logger.info("=" * 70)
        logger.info("🚀 ENTERPRISE AI ORCHESTRATION SYSTEM")
        logger.info("=" * 70)
        
        try:
            # Step 1: Start vLLM Server
            logger.info("\n[1/5] Launching vLLM Server...")
            if not self.vllm_manager.start_server():
                raise RuntimeError("vLLM server failed to start")
            
            # Verify vLLM health
            vllm_health = await check_http_endpoint(
                f"http://{config.VLLM_HOST}:{config.VLLM_PORT}/health",
                timeout=config.VLLM_STARTUP_TIMEOUT
            )
            if not vllm_health:
                logger.warning("vLLM health check failed, but continuing...")
            
            # Step 2: Start FastAPI
            logger.info("\n[2/5] Launching FastAPI Service...")
            if not self.api_manager.start_server():
                raise RuntimeError("FastAPI server failed to start")
            
            # Verify FastAPI health
            api_health = await check_http_endpoint(
                f"http://{config.API_HOST}:{config.API_PORT}/health",
                timeout=config.API_STARTUP_TIMEOUT
            )
            if not api_health:
                logger.warning("FastAPI health check failed, but continuing...")
            
            # Step 3: Start Dashboard
            logger.info("\n[3/5] Launching Dashboard...")
            if not self.dashboard_manager.start_server():
                logger.warning("Dashboard failed to start, but continuing...")
            
            # Step 4: Setup NGROK
            logger.info("\n[4/5] Configuring NGROK...")
            if not self.ngrok_manager.setup_auth():
                logger.warning("NGROK setup failed - services available on localhost only")
                self.print_local_urls()
                return
            
            # Step 5: Create Tunnels (only after services are confirmed running)
            logger.info("\n[5/5] Creating Public Tunnels...")
            
            api_tunnel = self.ngrok_manager.create_tunnel(config.API_PORT, "api")
            dashboard_tunnel = self.ngrok_manager.create_tunnel(config.DASHBOARD_PORT, "dashboard")
            
            # Print URLs
            logger.info("\n" + "=" * 70)
            logger.success("✅ ALL SERVICES RUNNING")
            logger.info("=" * 70)
            
            if api_tunnel:
                logger.info(f"\n🌐 PUBLIC URLS:")
                logger.info(f"   📡 API Endpoint:    {api_tunnel}")
                logger.info(f"   📊 Dashboard:       {dashboard_tunnel}/dashboard.html")
                logger.info(f"   📚 API Docs:        {api_tunnel}/docs")
                logger.info(f"   🔍 Health Check:    {api_tunnel}/health")
            
            logger.info(f"\n🏠 LOCAL URLS:")
            self.print_local_urls()
            
            logger.info("\n" + "=" * 70)
            logger.info("🔄 System is running. Press Ctrl+C to stop...")
            logger.info("=" * 70 + "\n")
            
            # Start health monitoring
            await self.health_check_loop()
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def print_local_urls(self):
        """Print local access URLs"""
        logger.info(f"   📡 API:             http://{config.API_HOST}:{config.API_PORT}")
        logger.info(f"   📊 Dashboard:       http://{config.API_HOST}:{config.DASHBOARD_PORT}/dashboard.html")
        logger.info(f"   🤖 vLLM:            http://{config.VLLM_HOST}:{config.VLLM_PORT}")
    
    def cleanup(self):
        """Cleanup all resources"""
        logger.info("\n🧹 Cleaning up...")
        self.ngrok_manager.cleanup()
        logger.success("👋 Shutdown complete!")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    orchestrator = SystemOrchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        sys.exit(1)
    
    # Run orchestrator
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
