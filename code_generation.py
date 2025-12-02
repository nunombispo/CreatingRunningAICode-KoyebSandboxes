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
        "IMPORTANT: Provide ONLY the code without any explanations, markdown formatting, "
        "code blocks, or additional text. Return pure code that can be executed directly."
    )
    
    payload = {
        "model": model,
        "prompt": enhanced_prompt,
        "stream": False
    }
    
    print(f"Generating code with model '{model}'...")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print("This may take a while...")
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("response", "").strip()
        
        if not generated_text:
            raise ValueError("Empty response from Ollama")
        
        # Extract code from markdown if present
        code = extract_code_from_markdown(generated_text)
        
        if not code:
            raise ValueError("No code extracted from response")
        
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


def save_code_to_file(code: str, output_file: str) -> None:
    """
    Save generated code to a file.
    
    Args:
        code: The code to save
        output_file: Path to the output file
    """
    output_path = Path(output_file)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write code to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"✓ Code saved to: {output_path.absolute()}")
    print(f"  File size: {len(code)} characters ({len(code.splitlines())} lines)")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Generate code using Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s llama3.2 "Write a Python function to calculate factorial" output.py
  %(prog)s codellama "Create a REST API endpoint" api.py --host http://localhost:11434
  %(prog)s deepseek-coder "Generate a sorting algorithm" sort.py --timeout 600
        """
    )
    
    parser.add_argument(
        "model",
        help="Ollama model name (e.g., llama3.2, codellama, deepseek-coder)"
    )
    
    parser.add_argument(
        "prompt",
        help="Prompt describing the code to generate"
    )
    
    parser.add_argument(
        "output",
        help="Output filename for the generated code"
    )
    
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama API host URL (default: http://127.0.0.1:11434)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate code
        code = generate_code_with_ollama(
            model=args.model,
            prompt=args.prompt,
            ollama_host=args.host,
            timeout=args.timeout
        )
        
        # Save to file
        save_code_to_file(code, args.output)
        
        print("\n✓ Code generation completed successfully!")
        return 0
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ Connection Error: {e}", file=sys.stderr)
        print("\nMake sure Ollama is running. You can start it with:", file=sys.stderr)
        print("  ollama serve", file=sys.stderr)
        return 1
        
    except requests.exceptions.Timeout as e:
        print(f"\n✗ Timeout Error: {e}", file=sys.stderr)
        print(f"\nTry increasing the timeout with --timeout option", file=sys.stderr)
        return 1
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request Error: {e}", file=sys.stderr)
        return 1
        
    except ValueError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        return 130
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

