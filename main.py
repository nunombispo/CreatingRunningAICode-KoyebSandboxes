"""
AI Code Generation and Execution with Koyeb Sandboxes

This application uses Ollama to generate code with multiple AI models
and executes the generated code securely in isolated GPU-enabled Koyeb sandboxes.

All code generation happens inside the sandbox for maximum security and isolation.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from koyeb import Sandbox

# Load environment variables
load_dotenv()

# Get API token and remove quotes
api_token = os.getenv("KOYEB_API_TOKEN")
api_token = api_token.replace('"', '')

# Configure logging to file and terminal with timestamp
log_filename = f"sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),  # Also log to terminal
    ]
)
logger = logging.getLogger(__name__)


# Class to manage AI code generation and execution in GPU-enabled Koyeb sandboxes
class AICodeSandbox:
    """Manages AI code generation and execution in GPU-enabled Koyeb sandboxes."""
    
    # Constructor
    def __init__(self, api_token: str, use_gpu: bool = True):
        """
        Initialize the AICodeSandbox.
        
        Args:
            api_token: Koyeb API token.
            use_gpu: Whether to request GPU-enabled sandbox instances.
            models: List of models to use.
        """

        self.api_token = api_token
        if not self.api_token:
            raise ValueError(
                "Koyeb API token is required."
            )
        
        self.use_gpu = use_gpu

    
    # Check if GPU is available in the sandbox
    def _check_gpu_in_sandbox(self, sandbox: Sandbox) -> bool:
        """
        Check if GPU is available in the sandbox.
        
        Args:
            sandbox: The sandbox instance
        
        Returns:
            True if GPU is available, False otherwise
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info("Checking GPU in sandbox...")
            logger.info("=" * 60)

            # Check for NVIDIA GPU
            result = sandbox.exec(
                "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no_gpu'",
                timeout=600
            )
            if "no_gpu" not in result.stdout and result.stdout.strip():
                logger.info(f"GPU detected: {result.stdout.strip()}")
                return True
            else:
                logger.info("No GPU detected (running on CPU)")
                return False
        except Exception as e:
            logger.error(f"Error checking GPU: {str(e)}")
            return False

    # Create a sandbox
    def _create_sandbox(self, gpu_instance_type: Optional[str] = None, region: Optional[str] = None):
        sandbox = None
        has_gpu = False
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info("Creating Koyeb sandbox...")
            logger.info("=" * 60)

            # Create GPU-enabled sandbox
            logger.info("Creating Koyeb sandbox with GPU support...")
            sandbox_params = {
                "image": "ubuntu:latest",
                "name": "ai-code-generation-gpu",
                "wait_ready": True,
                "timeout": 600,
                "api_token": self.api_token,
                "instance_type": "small",
            }
            
            # Specify GPU instance type and region
            if self.use_gpu:
                sandbox_params["instance_type"] = gpu_instance_type
                sandbox_params["region"] = region
            
            # Create the sandbox
            sandbox = Sandbox.create(**sandbox_params)
            logger.info(f"Sandbox created successfully (ID: {sandbox.id})")
            
            # Check for GPU
            if self.use_gpu:
                has_gpu = self._check_gpu_in_sandbox(sandbox)
            else:
                has_gpu = False
            return sandbox, has_gpu
        except Exception as e:
            logger.error(f"Error creating sandbox: {str(e)}")
            return None, False

    # Install Ollama in the sandbox
    def _install_ollama_in_sandbox(self, sandbox: Sandbox) -> bool:
        """
        Install Ollama in the sandbox.
        
        Args:
            sandbox: The sandbox instance
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info("Installing Ollama in sandbox...")
            logger.info("=" * 60)
            
            # Install required packages
            logger.info("Installing system packages...")
            result = sandbox.exec(
                "apt-get update -qq && apt-get install -y -qq curl procps lshw python3 python3-pip python3-requests",
                on_stdout=lambda data: logger.info(data.strip()),
                on_stderr=lambda data: logger.error(data.strip())
            )
            if result.exit_code != 0:
                logger.error("Failed to install required packages")
                return False
            
            # Download and install Ollama
            logger.info("Downloading and installing Ollama...")
            result = sandbox.exec(
                "curl -fsSL https://ollama.com/install.sh | sh",
                on_stdout=lambda data: logger.info(data.strip()),
                on_stderr=lambda data: logger.error(data.strip())
            )
            if result.exit_code != 0:
                logger.error(f"Failed to install Ollama: {result.stderr}")
                return False
            
            # Start Ollama service in background
            logger.info("Starting Ollama service...")
            sandbox.launch_process("ollama serve")
            
            # Wait for Ollama to start
            max_retries = 10
            for i in range(max_retries):
                time.sleep(1)
                result = sandbox.exec("ollama list")
                if result.exit_code == 0:
                    logger.info("Ollama started successfully")
                    return True
                else:
                    logger.error(f"Ollama failed to start: {result.stderr}")
                    return False
            logger.error(f"Ollama failed to start after {max_retries} retries")
            return False
        except Exception as e:
            logger.error(f"Error installing Ollama: {str(e)}")
            return False

    # Pull a model in the sandbox
    def _pull_model_in_sandbox(self, sandbox: Sandbox, model) -> bool:
        """
        Pull a model in the sandbox.
        
        Args:
            sandbox: The sandbox instance
            model: Model to pull
        
        Returns:
            True if pulling model successful, False otherwise
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info(f"Pulling models in sandbox...")
            logger.info("=" * 60)
            
            # Pull the model
            logger.info(f"Pulling model {model} in sandbox...")
            result = sandbox.exec(f"ollama pull {model}", 
                timeout=600, 
                on_stdout=lambda data: logger.info(data.strip()), 
                on_stderr=lambda data: logger.error(data.strip())
            )
            if result.exit_code == 0:
                logger.info(f"Model {model} pulled successfully")
                return True
            else:
                logger.error(f"Failed to pull model {model}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model {model}: {str(e)}")
            return False

    # Generate code in the sandbox
    def _generate_code_in_sandbox(self, sandbox: Sandbox, model: str, prompt: str, output: str) -> (bool, str):
        """
        Generate code in the sandbox.
        
        Args:
            sandbox: The sandbox instance
            model: Model to use
            prompt: Prompt to send to the model
            output: Output file to save the generated code

        Returns:
            True if generation successful, False otherwise
            Filename of the generated code
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info("Generating code in sandbox...")
            logger.info("=" * 60)
            
            # Copy file code_generation.py to sandbox
            fs = sandbox.filesystem
            fs.upload_file("code_generation.py", "/tmp/code_generation.py")
            logger.info(f"Code generation.py uploaded to sandbox")
            
            # Execute code_generation.py
            filename = f"/tmp/{model}-{output}"
            logger.info(f"Executing code_generation.py with model {model}, prompt {prompt}, and output {filename}")
            result = sandbox.exec(f"python3 /tmp/code_generation.py {model} \"{prompt}\" \"{filename}\"", 
                timeout=600,
                on_stdout=lambda data: logger.info(data.strip()), 
                on_stderr=lambda data: logger.error(data.strip())
            )
            if result.exit_code == 0:
                logger.info(f"Code generated successfully")
                return True, filename
            else:
                logger.error(f"Failed to generate code: {result.stderr}")
                return False, ""
        except Exception as e:
            logger.error(f"Error generating code with {model}: {str(e)}")
            return False, ""

    # Execute code in the sandbox
    def _execute_code_in_sandbox(self, sandbox: Sandbox, filename: str) -> bool:
        """
        Execute code in the sandbox.
        
        Args:
            sandbox: The sandbox instance
            filename: Filename of the code to execute
        
        Returns:
            True if execution successful, False otherwise
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info("Executing code in sandbox...")
            logger.info("=" * 60)

            # Print the code to execute
            fs = sandbox.filesystem
            code = fs.read_file(filename)
            logger.info(f"Code to execute:\n")
            logger.info("-" * 60)
            for line in code:
                logger.info(line.strip())
            logger.info("-" * 60)
            # Execute the code
            logger.info(f"Executing code in sandbox, file: {filename}...")
            result = sandbox.exec(f"python3 {filename}", 
                timeout=600,
                on_stdout=lambda data: logger.info(data.strip()), 
                on_stderr=lambda data: logger.error(data.strip())
            )
            if result.exit_code == 0:
                logger.info(f"Code executed successfully")
                return True
            else:
                logger.error(f"Failed to execute code: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return False
    
    
    # Delete the sandbox
    def _delete_sandbox(self, sandbox: Sandbox) -> bool:
        """
        Delete the sandbox.
        
        Args:
            sandbox: The sandbox instance
        
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            logger.info("\n")
            logger.info("=" * 60)
            logger.info(f"Deleting sandbox...")
            logger.info("=" * 60)

            # Delete the sandbox
            logger.info(f"Deleting sandbox {sandbox.id}...")
            sandbox.delete()
            logger.info(f"Sandbox {sandbox.id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting sandbox {sandbox.id}: {str(e)}")
            return False

# Pipeline to generate code and execute it
def pipeline():
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("Starting AI Code Generation and Execution with Koyeb Sandboxes")
    logger.info("=" * 60)
    sandbox_manager = AICodeSandbox(api_token, use_gpu=True)
    gpu_instance_type = "gpu-nvidia-rtx-4000-sff-ada"
    region = "fra"
    sandbox, has_gpu = sandbox_manager._create_sandbox(gpu_instance_type, region)
    if sandbox and has_gpu:
        if sandbox_manager._install_ollama_in_sandbox(sandbox):
            models = ["llama3.2", "codellama", "deepseek-coder"]
            for model in models:
                if sandbox_manager._pull_model_in_sandbox(sandbox, model):
                    prompt = "Write a Python program to calculate factorial of n=5"
                    output = "output.py"
                    success, filename = sandbox_manager._generate_code_in_sandbox(sandbox, model, prompt, output)
                    if success:
                        sandbox_manager._execute_code_in_sandbox(sandbox, filename)
            # Delete the sandbox
            sandbox_manager._delete_sandbox(sandbox)
                          

if __name__ == "__main__":
    pipeline()