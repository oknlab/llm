#!/usr/bin/env python3
"""
Google Colab launcher with NGROK tunneling for AI Agents Platform.
Optimized for serverless deployment with minimal overhead.
"""

import os
import sys
import subprocess
import threading
import json
import asyncio
from pathlib import Path

class ColabLauncher:
    def __init__(self):
        self.ngrok_auth = os.getenv('NGROK_AUTH_TOKEN')
        self.api_port = int(os.getenv('API_PORT', 8000))
        self.dashboard_port = int(os.getenv('DASHBOARD_PORT', 8080))
        
    def setup_environment(self):
        """Configure Colab environment with optimized settings."""
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])
        
        if self.ngrok_auth:
            subprocess.run(['ngrok', 'authtoken', self.ngrok_auth], check=False)
    
    def start_ngrok(self, port, name):
        """Initialize NGROK tunnel for external access."""
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port, bind_tls=True)
            print(f"[{name}] Public URL: {public_url}")
            return public_url
        except Exception as e:
            print(f"NGROK setup failed: {e}")
            return None
    
    def launch_services(self):
        """Launch core services with threading for concurrent execution."""
        services = []
        
        # API Server
        api_thread = threading.Thread(
            target=lambda: subprocess.run([
                sys.executable, '-m', 'uvicorn', 
                'system:app', '--host', '0.0.0.0', 
                '--port', str(self.api_port), '--reload'
            ])
        )
        api_thread.daemon = True
        api_thread.start()
        services.append(('API', self.api_port))
        
        # Dashboard Server
        dashboard_thread = threading.Thread(
            target=lambda: subprocess.run([
                sys.executable, '-m', 'http.server', 
                str(self.dashboard_port), '--directory', '.'
            ])
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        services.append(('Dashboard', self.dashboard_port))
        
        # NGROK tunnels
        for name, port in services:
            self.start_ngrok(port, name)
        
        print("\n✓ Platform initialized. Services running.")
        
    def run(self):
        """Main execution flow."""
        self.setup_environment()
        self.launch_services()
        
        try:
            while True:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            print("\nShutting down platform...")
            sys.exit(0)

if __name__ == "__main__":
    launcher = ColabLauncher()
    launcher.run()
