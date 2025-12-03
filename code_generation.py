#!/usr/bin/env python3
"""
Code Generation Tool using Ollama

This application generates code using Ollama models based on a prompt
and saves the generated code to a file.
"""

import argparse
import sys
import re
import requests
from pathlib import Path

# Extract code from markdown
def extract_code_from_markdown(text: str) -> str:
    """
    Extract code from markdown code blocks or return text as-is.
    
    Args:
        text: Text that may contain markdown code blocks
        
    Returns:
        Extracted code without markdown formatting
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Try to find code blocks with language tags (```python, ```bash, etc.)
    code_block_pattern = r'```(?:\w+)?\n?(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # Try to find code blocks without language tags
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    
    # If no code blocks, return as-is (might be plain code)
    return text


# Generate code with Ollama
def generate_code_with_ollama(
    model: str,
    prompt: str,
    ollama_host: str = "http://127.0.0.1:11434",
    timeout: int = 300
) -> str:
    """
    Generate code using Ollama API.
    
    Args:
        model: Ollama model name to use
        prompt: The prompt for code generation
        ollama_host: Ollama API host URL (default: http://127.0.0.1:11434)
        timeout: Request timeout in seconds (default: 300)
        
    Returns:
        Generated code as a string
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
        ValueError: If the response is empty or invalid
    """
    url = f"{ollama_host}/api/generate"
    
    # Enhanced prompt with clear instructions
    enhanced_prompt = (
        f"{prompt}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Output ONLY executable code\n"
        "- Do NOT include markdown code blocks (no ``` or ```)\n"
        "- Do NOT include explanations, comments about the code, or any text before/after\n"
        "- Do NOT include language tags or formatting\n"
        "- Return pure, runnable code starting from the first line\n"
        "Your response must be code that can be executed directly without any modifications."
    )
    
    # Define the payload for the API request
    payload = {
        "model": model,
        "prompt": enhanced_prompt,
        "stream": False
    }
    
    # Print the prompt and model
    print(f"Generating code with model '{model}'...\n")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}...\n")
    print("This may take a while...\n")
    

    try:
        # Make the API request
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Get the response
        result = response.json()
        generated_text = result.get("response", "").strip()
        
        # Check if the response is empty
        if not generated_text:
            raise ValueError("Empty response from Ollama")
        
        # Extract code from the response
        code = extract_code_from_markdown(generated_text)
        
        # Check if the code is empty
        if not code:
            raise ValueError("No code extracted from response")
        
        # Return the code
        return code
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            f"Request timed out after {timeout} seconds. "
            "The model may be too slow or the prompt too complex."
        )   
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError(
            f"Could not connect to Ollama at {ollama_host}. "
            "Make sure Ollama is running and accessible."
        )
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Request failed: {str(e)}")


# Save code to a file
def save_code_to_file(code: str, output_file: str) -> None:
    """
    Save generated code to a file.
    
    Args:
        code: The code to save
        output_file: Path to the output file
    """
    # Create the output path
    output_path = Path(output_file)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the code to the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Print the output path and file size
    print(f"Code saved to: {output_path.absolute()}")
    print(f"File size: {len(code)} characters ({len(code.splitlines())} lines)")

def main():
    """Main entry point for the application."""
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Generate code using Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter

    )
    
    # Add the model argument
    parser.add_argument(
        "model",
        help="Ollama model name (e.g., llama3.2, codellama, deepseek-coder)"
    )
    
    # Add the prompt argument
    parser.add_argument(
        "prompt",
        help="Prompt describing the code to generate"
    )
    
    # Add the output argument
    parser.add_argument(
        "output",
        help="Output filename for the generated code"
    )
    
    # Add the host argument
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama API host URL (default: http://127.0.0.1:11434)"
    )
    
    # Add the timeout argument
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    try:
        # Generate the code
        code = generate_code_with_ollama(
            model=args.model,
            prompt=args.prompt,
            ollama_host=args.host,
            timeout=args.timeout
        )
        
        # Save to file
        save_code_to_file(code, args.output)
        
        # Print the success message and return the success code
        print("Code generation completed successfully!")
        return 0
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        # Return the error code
        return 1
        
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error: {e}", file=sys.stderr)
        # Return the error code
        return 1
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}", file=sys.stderr)
        # Return the error code
        return 1
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        # Return the error code
        return 1
        
    except KeyboardInterrupt:
        print("Operation cancelled by user.", file=sys.stderr)
        # Return the error code
        return 130
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        # Return the error code
        return 1


if __name__ == "__main__":
    # Exit the program with the return code
    sys.exit(main())

