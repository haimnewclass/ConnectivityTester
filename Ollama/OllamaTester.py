import requests
import socket
import time
import platform
import subprocess
from typing import Dict, Optional, Tuple
from enum import Enum
import logging
from urllib.parse import urlparse

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    NO_INTERNET = "no_internet"
    SERVER_UNREACHABLE = "server_unreachable"
    MODEL_NOT_LOADED = "model_not_loaded"
    AUTHENTICATION_FAILED = "authentication_failed"
    TIMEOUT = "timeout"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    REMOTE_SERVER_UNREACHABLE = "remote_server_unreachable"
    UNKNOWN_ERROR = "unknown_error"

class OllamaTester:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        """
        Initialize the OllamaTester with configuration parameters.
        
        Args:
            base_url (str): The base URL of the Ollama server
            timeout (int): Timeout in seconds for requests
        """
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.is_remote = not self._is_local_url(base_url)
        
    def _is_local_url(self, url: str) -> bool:
        """
        Check if the URL is local or remote.
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if URL is local, False if remote
        """
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if hostname in ['localhost', '127.0.0.1']:
            return True
            
        try:
            # Check if the hostname resolves to a local IP
            ip = socket.gethostbyname(hostname)
            return ip.startswith(('127.', '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.'))
        except socket.gaierror:
            return False
            
    def ping_host(self, host: str, count: int = 4) -> Tuple[bool, str]:
        """
        Ping a host to check connectivity.
        
        Args:
            host (str): Host to ping
            count (int): Number of ping attempts
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Extract hostname from URL if needed
            if host.startswith(('http://', 'https://')):
                host = urlparse(host).hostname
                
            # Different ping commands for different operating systems
            if platform.system().lower() == "windows":
                ping_cmd = ["ping", "-n", str(count), host]
            else:
                ping_cmd = ["ping", "-c", str(count), host]
                
            result = subprocess.run(
                ping_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, f"Successfully pinged {host}"
            return False, f"Failed to ping {host}: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            return False, f"Ping timeout for {host}"
        except Exception as e:
            return False, f"Error pinging {host}: {str(e)}"
            
    def check_remote_server(self) -> Tuple[bool, str]:
        """
        Check remote server connectivity with ping test.
        
        Returns:
            Tuple[bool, str]: (is_available, status_message)
        """
        if not self.is_remote:
            return True, "Local server, no ping test needed"
            
        host = urlparse(self.base_url).hostname
        ping_success, ping_message = self.ping_host(host)
        
        if not ping_success:
            return False, f"Remote server ping failed: {ping_message}"
            
        return True, "Remote server is reachable"
        
    def check_internet_connectivity(self) -> bool:
        """
        Check if there is general internet connectivity.
        
        Returns:
            bool: True if internet is available, False otherwise
        """
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
            
    def check_server_availability(self) -> Tuple[bool, str]:
        """
        Check if the Ollama server is available and responding.
        
        Returns:
            Tuple[bool, str]: (is_available, status_message)
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                return True, "Server is available and responding"
            return False, f"Server returned status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Server is not reachable: {str(e)}"
            
    def check_model_status(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if a specific model is loaded and available.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            Tuple[bool, str]: (is_available, status_message)
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model.get("name") == model_name:
                        return True, f"Model {model_name} is available"
                return False, f"Model {model_name} is not loaded"
            return False, f"Failed to check model status: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Error checking model status: {str(e)}"
            
    def test_connection(self) -> Dict[str, any]:
        """
        Perform a comprehensive connection test.
        
        Returns:
            Dict[str, any]: Dictionary containing test results and status
        """
        results = {
            "timestamp": time.time(),
            "status": ConnectionStatus.UNKNOWN_ERROR,
            "details": {}
        }
        
        # Check internet connectivity
        has_internet = self.check_internet_connectivity()
        results["details"]["internet"] = {
            "available": has_internet,
            "message": "Internet is available" if has_internet else "No internet connection"
        }
        
        if not has_internet:
            results["status"] = ConnectionStatus.NO_INTERNET
            return results
            
        # Check remote server if applicable
        if self.is_remote:
            remote_available, remote_message = self.check_remote_server()
            results["details"]["remote"] = {
                "available": remote_available,
                "message": remote_message
            }
            
            if not remote_available:
                results["status"] = ConnectionStatus.REMOTE_SERVER_UNREACHABLE
                return results
                
        # Check server availability
        server_available, server_message = self.check_server_availability()
        results["details"]["server"] = {
            "available": server_available,
            "message": server_message
        }
        
        if not server_available:
            results["status"] = ConnectionStatus.SERVER_UNREACHABLE
            return results
            
        # Check default model availability
        model_available, model_message = self.check_model_status("llama2")
        results["details"]["model"] = {
            "available": model_available,
            "message": model_message
        }
        
        if not model_available:
            results["status"] = ConnectionStatus.MODEL_NOT_LOADED
            return results
            
        results["status"] = ConnectionStatus.CONNECTED
        return results
        
    def test_endpoint(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Test a specific API endpoint.
        
        Args:
            endpoint (str): API endpoint to test
            method (str): HTTP method to use
            data (Optional[Dict]): Data to send with the request
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(
                method=method,
                url=url,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 401:
                return False, "Authentication failed"
            elif response.status_code == 404:
                return False, "Endpoint not found"
            elif response.status_code >= 500:
                return False, f"Server error: {response.status_code}"
            elif response.status_code >= 400:
                return False, f"Request failed: {response.status_code}"
                
            return True, f"Endpoint test successful: {response.status_code}"
            
        except requests.exceptions.Timeout:
            return False, "Request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Request failed: {str(e)}"
            
    def get_detailed_status(self) -> Dict[str, any]:
        """
        Get a detailed status report of the Ollama service.
        
        Returns:
            Dict[str, any]: Detailed status information
        """
        status = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "timeout": self.timeout,
            "is_remote": self.is_remote,
            "tests": {}
        }
        
        # Test basic connectivity
        status["tests"]["internet"] = self.check_internet_connectivity()
        
        # Test remote server if applicable
        if self.is_remote:
            remote_available, remote_message = self.check_remote_server()
            status["tests"]["remote"] = {
                "available": remote_available,
                "message": remote_message
            }
        
        # Test server endpoints
        endpoints = [
            "/api/tags",
            "/api/version",
            "/api/health"
        ]
        
        for endpoint in endpoints:
            success, message = self.test_endpoint(endpoint)
            status["tests"][endpoint] = {
                "success": success,
                "message": message
            }
            
        return status 