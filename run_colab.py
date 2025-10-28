#!/usr/bin/env python3
"""
Google Colab launcher with SINGLE NGROK tunnel for both API and Dashboard.
Unified access through one URL.
"""

import os
import sys
import subprocess
import threading
import json
import socket
import time
from pathlib import Path

class ColabLauncher:
    def __init__(self):
        self.ngrok_auth = os.getenv('NGROK_AUTH_TOKEN')
        self.unified_port = self._find_free_port(8000)
        os.environ['UNIFIED_PORT'] = str(self.unified_port)
        
    def _find_free_port(self, start_port):
        """Find next available port starting from start_port."""
        port = start_port
        while port < start_port + 100:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    port += 1
        return start_port
    
    def setup_environment(self):
        """Configure environment with proper dependency installation."""
        print("🔧 Installing dependencies...")
        
        # Install requirements with proper error handling
        req_file = Path('requirements.txt')
        if req_file.exists():
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"⚠️ Warning: Some packages failed to install: {result.stderr}")
        
        # Install pyngrok separately if needed
        try:
            import pyngrok
        except ImportError:
            print("📦 Installing pyngrok...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok'])
        
        # Setup NGROK
        if self.ngrok_auth:
            try:
                from pyngrok import ngrok
                ngrok.set_auth_token(self.ngrok_auth)
                print("✅ NGROK configured successfully")
            except Exception as e:
                print(f"⚠️ NGROK setup warning: {e}")
        else:
            print("ℹ️ No NGROK token provided - will run locally only")
    
    def start_unified_tunnel(self):
        """Start single NGROK tunnel for unified service."""
        try:
            from pyngrok import ngrok
            
            # Kill any existing tunnels
            ngrok.kill()
            
            # Create single tunnel
            public_url = ngrok.connect(self.unified_port, bind_tls=True)
            
            return str(public_url)
        except ImportError:
            print("⚠️ NGROK not available - using local access only")
            return f"http://localhost:{self.unified_port}"
        except Exception as e:
            print(f"⚠️ NGROK tunnel failed: {e}")
            return f"http://localhost:{self.unified_port}"
    
    def launch_unified_service(self):
        """Launch unified API + Dashboard service."""
        print(f"\n🚀 Starting unified service on port {self.unified_port}...")
        
        # Start unified FastAPI server
        server_thread = threading.Thread(
            target=lambda: subprocess.run([
                sys.executable, '-m', 'uvicorn', 
                'system:app', '--host', '0.0.0.0', 
                '--port', str(self.unified_port), '--reload'
            ])
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for service to start
        time.sleep(5)
        
        # Start single NGROK tunnel
        public_url = self.start_unified_tunnel()
        
        # Display access information
        self.display_access_info(public_url)
        
        return public_url
    
    def display_access_info(self, public_url):
        """Display formatted access information."""
        print("\n" + "="*60)
        print("🎯 AI AGENTS PLATFORM - UNIFIED ACCESS")
        print("="*60)
        
        local_url = f"http://localhost:{self.unified_port}"
        
        if 'ngrok' in public_url:
            print(f"""
📡 PUBLIC ACCESS (Single URL for Everything):
   {public_url}
   
   • Dashboard: {public_url}/
   • API Docs:  {public_url}/docs
   • Health:    {public_url}/health
            """)
        
        print(f"""
💻 LOCAL ACCESS:
   {local_url}
   
   • Dashboard: {local_url}/
   • API Docs:  {local_url}/docs
   • Health:    {local_url}/health
        """)
        
        print("="*60)
        print("✅ Platform Ready! Access everything through ONE URL")
        print("="*60 + "\n")
    
    def run(self):
        """Main execution flow."""
        print("""
╔══════════════════════════════════════════════════════════╗
║     🤖 AI AGENTS PLATFORM - Enterprise Edition 🤖        ║
║         Single URL Access - Unified Interface            ║
╚══════════════════════════════════════════════════════════╝
        """)
        
        self.setup_environment()
        public_url = self.launch_unified_service()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Shutting down platform...")
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except:
                pass
            sys.exit(0)

if __name__ == "__main__":
    launcher = ColabLauncher()
    launcher.run()
