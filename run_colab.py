#!/usr/bin/env python3
"""
Google Colab launcher with NGROK tunneling for AI Agents Platform.
Fixed port conflicts and dependency handling.
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
        self.ngrok_auth = os.getenv('1vikehg18jsR9XrEzKEybCifEr9_AWWFzoCD58Xa151mXfLd')
        self.api_port = self._find_free_port(8000)
        self.dashboard_port = self._find_free_port(8080)
        os.environ['API_PORT'] = str(self.api_port)
        os.environ['DASHBOARD_PORT'] = str(self.dashboard_port)
        
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
        print("Installing dependencies...")
        
        # Install requirements with proper error handling
        req_file = Path('requirements.txt')
        if req_file.exists():
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: Some packages failed to install: {result.stderr}")
        
        # Install pyngrok separately if needed
        try:
            import pyngrok
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyngrok'])
        
        # Setup NGROK
        if self.ngrok_auth:
            try:
                from pyngrok import ngrok
                ngrok.set_auth_token(self.ngrok_auth)
                print("NGROK configured successfully")
            except Exception as e:
                print(f"NGROK setup warning: {e}")
    
    def start_ngrok(self, port, name):
        """Initialize NGROK tunnel with error handling."""
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port, bind_tls=True)
            print(f"\n✓ [{name}] Public URL: {public_url}")
            return public_url
        except ImportError:
            print(f"[{name}] NGROK not available - using local access only")
            return f"http://localhost:{port}"
        except Exception as e:
            print(f"[{name}] NGROK tunnel failed: {e}")
            return f"http://localhost:{port}"
    
    def serve_dashboard(self):
        """Custom dashboard server to avoid port conflicts."""
        import http.server
        import socketserver
        
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard.html':
                    self.path = '/dashboard.html'
                return super().do_GET()
        
        with socketserver.TCPServer(("", self.dashboard_port), DashboardHandler) as httpd:
            print(f"Dashboard serving at port {self.dashboard_port}")
            httpd.serve_forever()
    
    def launch_services(self):
        """Launch services with proper error handling."""
        print(f"\nStarting services...")
        print(f"API Port: {self.api_port}")
        print(f"Dashboard Port: {self.dashboard_port}")
        
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
        
        # Dashboard Server
        dashboard_thread = threading.Thread(target=self.serve_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Wait for services to start
        time.sleep(3)
        
        # Start NGROK tunnels
        api_url = self.start_ngrok(self.api_port, 'API')
        dashboard_url = self.start_ngrok(self.dashboard_port, 'Dashboard')
        
        print("\n" + "="*50)
        print("✓ AI AGENTS PLATFORM READY")
        print("="*50)
        print(f"API Local: http://localhost:{self.api_port}")
        print(f"Dashboard Local: http://localhost:{self.dashboard_port}")
        if 'ngrok' in str(api_url):
            print(f"API Public: {api_url}")
            print(f"Dashboard Public: {dashboard_url}")
        print("="*50 + "\n")
        
    def run(self):
        """Main execution flow."""
        self.setup_environment()
        self.launch_services()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down platform...")
            sys.exit(0)

if __name__ == "__main__":
    launcher = ColabLauncher()
    launcher.run()
