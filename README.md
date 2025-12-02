# Creating and Running AI-Generated Code Securely with Koyeb Sandboxes

This project demonstrates how to use **Ollama** to generate code with multiple AI models **inside GPU-enabled Koyeb Sandboxes** and execute that code securely in isolated environments. All code generation happens inside the sandbox for maximum security and isolation.

## Features

- 🤖 **Multiple AI Models**: Generate code using different Ollama models (llama3.2, codellama, deepseek-coder, etc.)
- 🚀 **GPU Acceleration**: Code generation runs in GPU-enabled Koyeb sandboxes for optimal performance
- 🔒 **Fully Isolated**: Both code generation and execution happen in isolated sandbox environments
- 🧹 **Automatic Cleanup**: Sandboxes are automatically deleted after execution
- 📊 **Results Comparison**: Compare outputs from different AI models side-by-side
- 🔐 **Secure by Design**: No local AI model execution - everything runs in cloud sandboxes

## Prerequisites

- A [Koyeb account](https://www.koyeb.com) (Starter, Pro, or Scale Plan)
- Python 3.8+ installed on your machine
- **Note**: You do NOT need Ollama installed locally - it runs inside the sandboxes!

## Setup

### 1. Clone or Navigate to the Project

```bash
cd CreatingRunningAICode-KoyebSandboxes
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

On some systems, you may need to use `python3`:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Get Your Koyeb API Token

1. Go to [Koyeb Settings](https://app.koyeb.com/settings/api)
2. Click the **API** tab
3. Click **Create API token**
4. Provide a name (e.g., "sandbox-quickstart") and description
5. Click **Create** and copy the token (you won't be able to see it again)

### 6. Set Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your Koyeb API token:

```
KOYEB_API_TOKEN=your_api_token_here
```

Alternatively, you can set it directly in your terminal:

**Windows (Command Prompt):**

```cmd
set KOYEB_API_TOKEN=your_api_token_here
```

**Windows (PowerShell):**

```powershell
$env:KOYEB_API_TOKEN="your_api_token_here"
```

**macOS/Linux:**

```bash
export KOYEB_API_TOKEN="your_api_token_here"
```

## Usage

### Basic Usage

Run with the default example prompt:

```bash
python main.py
```

### Custom Prompt

Provide your own prompt as command line arguments:

```bash
python main.py "Write a Python function that calculates prime numbers up to 100"
```

### Example Prompts

```bash
# Calculate factorial
python main.py "Write a Python function that calculates the factorial of a number and prints the result for n=5"

# Hello World with date
python main.py "Create a Python script that prints 'Hello, World!' and the current date"

# Fibonacci sequence
python main.py "Write a Python function that generates the first 10 Fibonacci numbers and prints them"
```

## How It Works

1. **GPU Sandbox Creation**: A GPU-enabled Koyeb sandbox is created for code generation
2. **Ollama Installation**: Ollama is automatically installed and started inside the sandbox
3. **Model Download**: Required AI models are pulled inside the sandbox
4. **Code Generation**: Your prompt is sent to multiple Ollama models running inside the sandbox (with GPU acceleration)
5. **Code Execution**: Each generated code is executed in a separate isolated sandbox
6. **Results Collection**: Outputs from each model are collected and displayed
7. **Cleanup**: All sandboxes are automatically deleted after execution

**Key Security Feature**: All AI model execution happens inside isolated sandboxes - your local machine never runs the models directly!

## Project Structure

```
.
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── .env.example        # Example environment variables
├── .env                # Your environment variables (create this)
└── README.md           # This file
```

## Customization

### Change Models

Edit the `models` list in `main.py`:

```python
self.models = [
    "llama3.2",
    "codellama",
    "deepseek-coder",
    "mistral",  # Add more models
]
```

### Change Language

Modify the `language` parameter when calling `run_full_pipeline`:

```python
results = sandbox_manager.run_full_pipeline(
    prompt=prompt,
    language="bash"  # or "javascript", "python", etc.
)
```

### Use Specific Models

Pass a custom list of models:

```python
results = sandbox_manager.run_full_pipeline(
    prompt=prompt,
    models=["llama3.2", "mistral"]
)
```

## GPU Support

The application attempts to use GPU-enabled sandboxes for faster code generation. GPU availability depends on:

- Your Koyeb account tier (Pro or Scale plans typically have GPU access)
- Regional GPU availability
- Koyeb's GPU instance availability

If GPU is not available, the sandbox will fall back to CPU execution (slower but functional).

### Checking GPU in Sandbox

The application automatically checks for GPU availability and will display a message if GPU is detected. You can also verify GPU support in your Koyeb account settings.

## Troubleshooting

### GPU Not Available

If you see "No GPU detected", the sandbox will still work but code generation will be slower. This is normal if:

- Your account doesn't have GPU access
- GPU instances are not available in your region
- You're on a Starter plan (GPU typically requires Pro/Scale)

### Koyeb API Token Error

Verify your API token is set correctly:

```bash
echo $KOYEB_API_TOKEN  # Linux/macOS
echo %KOYEB_API_TOKEN%  # Windows CMD
```

### Model Download Timeout

Models are automatically downloaded inside the sandbox. If a model fails to download:

- Check your internet connection
- Verify the model name is correct
- Larger models may take several minutes to download

### Sandbox Creation Timeout

Sandbox creation usually takes a few seconds. If it times out, check your Koyeb account status and API token permissions.

## Security Notes

- **Complete Isolation**: Both code generation and execution happen in isolated sandboxes
- **No Local AI Execution**: AI models never run on your local machine
- **Automatic Cleanup**: Sandboxes are automatically deleted after execution
- **Fresh Environments**: Each execution happens in a fresh, clean environment
- **No Persistent Storage**: No data persists between executions
- **Ephemeral Sandboxes**: Sandboxes cannot access your local system or other sandboxes

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [Koyeb Sandboxes Documentation](https://www.koyeb.com/docs/sandboxes/sandbox-quickstart)
- [Ollama Documentation](https://ollama.ai/docs)
- [Koyeb Python SDK](https://github.com/koyeb/koyeb-sdk-python)
