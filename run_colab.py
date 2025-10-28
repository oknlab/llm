"""
Google Colab Orchestration Script
Launches vLLM Server, FastAPI Service, NGROK Tunnel, and Dashboard
"""

import os
import sys
import asyncio
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional

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
    DASHBOARD_FILE = BASE_DIR / "dashboard.html"
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    VLLM_HOST = os.getenv("MODEL_SERVER_HOST", "0.0.0.0")
    VLLM_PORT = int(os.getenv("MODEL_SERVER_PORT", "8000"))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "7860"))
    
    # NGROK Configuration
    NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    NGROK_REGION = os.getenv("NGROK_REGION", "us")
    
    # vLLM Performance
    TENSOR_PARALLEL = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    GPU_MEMORY = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))


config = ColabConfig()


# ============================================================================
# PROCESS MANAGEMENT
# ============================================================================

class ProcessManager:
    """Manage subprocess lifecycle"""
    
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.tunnels: dict[str, str] = {}
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Graceful shutdown on signal"""
        logger.warning(f"Received signal {signum}, shutting down...")
        self.shutdown_all()
        sys.exit(0)
    
    def start_process(
        self, 
        name: str, 
        command: list[str], 
        env: Optional[dict] = None
    ) -> subprocess.Popen:
        """Start a subprocess"""
        logger.info(f"Starting {name}: {' '.join(command)}")
        
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            text=True,
            bufsize=1
        )
        
        self.processes[name] = process
        logger.success(f"{name} started (PID: {process.pid})")
        
        return process
    
    def is_alive(self, name: str) -> bool:
        """Check if process is running"""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None
    
    def shutdown_all(self):
        """Shutdown all processes"""
        logger.info("Shutting down all processes...")
        
        # Close NGROK tunnels
        ngrok.kill()
        
        # Terminate processes
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Terminating {name} (PID: {process.pid})")
                process.terminate()
                
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}")
                    process.kill()
        
        logger.success("All processes shutdown")


# ============================================================================
# VLLM SERVER LAUNCHER
# ============================================================================

class VLLMServer:
    """vLLM Inference Server Manager"""
    
    @staticmethod
    def check_gpu():
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.success(f"GPU available: {gpu_name}")
                return True
            else:
                logger.warning("No GPU detected, using CPU (slow)")
                return False
        except ImportError:
            logger.error("PyTorch not installed")
            return False
    
    @staticmethod
    def install_model():
        """Download model from Hugging Face"""
        logger.info(f"Checking model: {config.MODEL_NAME}")
        
        try:
            from huggingface_hub import snapshot_download
            
            cache_dir = Path.home() / ".cache" / "huggingface"
            model_path = snapshot_download(
                repo_id=config.MODEL_NAME,
                cache_dir=cache_dir
            )
            
            logger.success(f"Model cached at: {model_path}")
            return model_path
        
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise
    
    @staticmethod
    def launch(process_manager: ProcessManager):
        """Launch vLLM server"""
        
        # Check GPU
        has_gpu = VLLMServer.check_gpu()
        
        # Install model
        model_path = VLLMServer.install_model()
        
        # Build vLLM command
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.MODEL_NAME,
            "--host", config.VLLM_HOST,
            "--port", str(config.VLLM_PORT),
            "--tensor-parallel-size", str(config.TENSOR_PARALLEL),
            "--gpu-memory-utilization", str(config.GPU_MEMORY),
            "--max-model-len", str(config.MAX_MODEL_LEN),
            "--trust-remote-code"
        ]
        
        if not has_gpu:
            vllm_cmd.extend(["--device", "cpu"])
        
        # Start vLLM
        process_manager.start_process("vLLM", vllm_cmd)
        
        # Wait for server to be ready
        logger.info("Waiting for vLLM server to be ready...")
        time.sleep(30)  # Adjust based on model size
        
        # Health check
        import httpx
        try:
            response = httpx.get(f"http://{config.VLLM_HOST}:{config.VLLM_PORT}/health", timeout=10)
            if response.status_code == 200:
                logger.success("vLLM server is ready")
            else:
                logger.error(f"vLLM health check failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"vLLM health check error: {e}")


# ============================================================================
# NGROK TUNNEL SETUP
# ============================================================================

class NGROKManager:
    """NGROK Tunnel Manager"""
    
    @staticmethod
    def setup_auth():
        """Configure NGROK authentication"""
        if not config.NGROK_TOKEN:
            logger.error("NGROK_AUTH_TOKEN not set in .env file")
            raise ValueError("NGROK token required")
        
        conf.get_default().auth_token = config.NGROK_TOKEN
        conf.get_default().region = config.NGROK_REGION
        
        logger.success("NGROK authenticated")
    
    @staticmethod
    def create_tunnel(port: int, name: str = "api") -> str:
        """Create NGROK tunnel"""
        try:
            tunnel = ngrok.connect(
                port,
                bind_tls=True,
                name=name
            )
            
            public_url = tunnel.public_url
            logger.success(f"NGROK tunnel created: {public_url}")
            
            return public_url
        
        except Exception as e:
            logger.error(f"NGROK tunnel creation failed: {e}")
            raise


# ============================================================================
# DASHBOARD SERVER
# ============================================================================

def serve_dashboard(process_manager: ProcessManager):
    """Serve HTML dashboard"""
    
    if not config.DASHBOARD_FILE.exists():
        logger.error(f"Dashboard file not found: {config.DASHBOARD_FILE}")
        return
    
    # Simple HTTP server for dashboard
    dashboard_cmd = [
        "python", "-m", "http.server",
        "8080",
        "--directory", str(config.BASE_DIR)
    ]
    
    process_manager.start_process("Dashboard", dashboard_cmd)
    logger.success("Dashboard server started on port 8080")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

async def main():
    """
    Main orchestration flow:
    1. Initialize process manager
    2. Launch vLLM server
    3. Start FastAPI service
    4. Create NGROK tunnels
    5. Launch dashboard
    6. Monitor services
    """
    
    logger.info("=" * 70)
    logger.info("🚀 ENTERPRISE AI ORCHESTRATION SYSTEM - COLAB LAUNCHER")
    logger.info("=" * 70)
    
    process_manager = ProcessManager()
    
    try:
        # Step 1: Launch vLLM Server
        logger.info("\n[1/5] Launching vLLM Inference Server...")
        VLLMServer.launch(process_manager)
        
        # Step 2: Configure NGROK
        logger.info("\n[2/5] Configuring NGROK...")
        NGROKManager.setup_auth()
        
        # Step 3: Start FastAPI Service
        logger.info("\n[3/5] Starting FastAPI Service...")
        api_cmd = [
            "python", "-m", "uvicorn",
            "system:app",
            "--host", config.API_HOST,
            "--port", str(config.API_PORT),
            "--log-level", "info"
        ]
        process_manager.start_process("FastAPI", api_cmd)
        time.sleep(10)  # Wait for API to start
        
        # Step 4: Create NGROK Tunnels
        logger.info("\n[4/5] Creating NGROK Tunnels...")
        api_tunnel = NGROKManager.create_tunnel(config.API_PORT, "api")
        dashboard_tunnel = NGROKManager.create_tunnel(8080, "dashboard")
        
        process_manager.tunnels["api"] = api_tunnel
        process_manager.tunnels["dashboard"] = dashboard_tunnel
        
        # Step 5: Launch Dashboard
        logger.info("\n[5/5] Launching Dashboard...")
        serve_dashboard(process_manager)
        
        # Display access URLs
        logger.info("\n" + "=" * 70)
        logger.success("✅ ALL SERVICES RUNNING")
        logger.info("=" * 70)
        logger.info(f"\n📡 API Endpoint:       {api_tunnel}")
        logger.info(f"📊 Dashboard:          {dashboard_tunnel}/dashboard.html")
        logger.info(f"📚 API Docs:           {api_tunnel}/docs")
        logger.info(f"🔍 Health Check:       {api_tunnel}/health")
        logger.info("\n" + "=" * 70)
        
        # Monitor services
        logger.info("\n🔄 Monitoring services (Press Ctrl+C to stop)...\n")
        
        while True:
            await asyncio.sleep(10)
            
            # Check service health
            for name in ["vLLM", "FastAPI", "Dashboard"]:
                if not process_manager.is_alive(name):
                    logger.error(f"❌ {name} process died!")
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Shutdown requested by user")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        process_manager.shutdown_all()
        logger.info("👋 Goodbye!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Run main orchestration
    asyncio.run(main())
