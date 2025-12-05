# Creating and Running AI-Generated Code Securely with Koyeb Sandboxes

This project demonstrates how to use **Ollama** to generate code with multiple AI models **inside GPU-enabled Koyeb Sandboxes** and execute that code securely in isolated environments.

All code generation happens inside the sandbox for maximum security and isolation.

## Features

- 🤖 **Multiple AI Models**: Generate code using different Ollama models (llama3.2, codellama, deepseek-coder, etc.)
- 🚀 **GPU Acceleration**: Code generation runs in GPU-enabled Koyeb sandboxes for optimal performance
- 🔒 **Fully Isolated**: Both code generation and execution happen in isolated sandbox environments
- 🧹 **Automatic Cleanup**: Sandboxes are automatically deleted after execution
- ⚙️ **Command-Line Interface**: Customize all settings via command-line arguments without editing code

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

## Usage

### Basic Usage

Run the pipeline with default settings:

```bash
python main.py
```

The default configuration:

- **Models**: llama3.2, codellama, deepseek-coder
- **Prompt**: "Write a Python program to calculate factorial of n=5. It should use a function."
- **Output**: output.py
- **GPU**: Enabled (requires GPU)
- **Instance Type**: gpu-nvidia-rtx-4000-sff-ada
- **Region**: fra

### Command-Line Options

Customize the pipeline using command-line arguments:

```bash
# View all available options
python main.py --help
```

#### Available Arguments

- `--models MODEL [MODEL ...]`: Specify AI models to use

  ```bash
  python main.py --models llama3.2 codellama
  python main.py --models deepseek-coder mistral
  ```

- `--prompt PROMPT`: Custom code generation prompt

  ```bash
  python main.py --prompt "Write a Python function to calculate prime numbers up to 100"
  ```

- `--output FILENAME`: Output filename for generated code

  ```bash
  python main.py --output my_code.py
  ```

- `--instance-type TYPE`: GPU instance type

  ```bash
  python main.py --instance-type gpu-nvidia-rtx-4000-sff-ada
  python main.py --instance-type gpu-nvidia-a100
  ```

- `--region REGION`: Koyeb region

  ```bash
  python main.py --region fra
  python main.py --region nyc
  python main.py --region sfo
  ```

- `--no-gpu`: Use CPU-only sandbox (disable GPU)

  ```bash
  python main.py --no-gpu
  ```

- `--no-require-gpu`: Allow CPU fallback if GPU is not available
  ```bash
  python main.py --no-require-gpu
  ```

### Usage Examples

```bash
# Default run
python main.py

# Custom prompt
python main.py --prompt "Create a REST API endpoint for user authentication"

# Multiple models with custom prompt
python main.py --models llama3.2 codellama --prompt "Write a sorting algorithm"

# Custom instance type and region
python main.py --instance-type gpu-nvidia-rtx-4000-sff-ada --region fra

# Allow CPU fallback if GPU unavailable
python main.py --no-require-gpu

# Use CPU-only sandbox
python main.py --no-gpu --prompt "Simple Python script"

# Combine multiple options
python main.py \
  --models llama3.2 deepseek-coder \
  --prompt "Write a Python class for managing a todo list" \
  --output todo_manager.py \
  --region nyc \
  --no-require-gpu
```

## How It Works

1. **Sandbox Creation**: A GPU-enabled Koyeb sandbox is created for code generation
2. **GPU Detection**: The system checks for GPU availability and logs the result
3. **Ollama Installation**: Ollama is automatically installed and started inside the sandbox
4. **Model Download**: Required AI models are pulled inside the sandbox (with progress logging)
5. **Code Generation**: Your prompt is sent to each Ollama model running inside the sandbox
   - The `code_generation.py` script is uploaded to the sandbox
   - Each model generates code based on your prompt
   - Generated code is saved to separate files per model
6. **Code Execution**: Each generated code is executed in the same sandbox
   - Code is displayed before execution
   - Execution results are logged in real-time
7. **Statistics & Summary**: Pipeline provides statistics on:
   - Models successfully pulled
   - Code files generated
   - Code executions completed
   - Errors encountered
8. **Automatic Cleanup**: Sandbox is automatically deleted after execution (even on errors)

**Key Security Feature**: All AI model execution happens inside isolated sandboxes - your local machine never runs the models directly!

**Logging**: All operations are logged to both:

- Console output (real-time)
- Log file: `sandbox_YYYYMMDD_HHMMSS.log` (timestamped)

## Project Structure

```
.
├── main.py                 # Main application and pipeline
├── code_generation.py       # Code generation script (uploaded to sandbox)
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .env                   # Your environment variables (create this)
└── README.md              # This file
```

### Key Files

- **`main.py`**: Main application that manages sandboxes, installs Ollama, pulls models, generates code, and executes it
- **`code_generation.py`**: Script that runs inside the sandbox to generate code using Ollama models
- **`requirements.txt`**: Python dependencies (koyeb-sdk, python-dotenv)

## Advanced Features

### Error Handling

The pipeline includes robust error handling:

- **Fault Tolerance**: If one model fails, others continue processing
- **Automatic Cleanup**: Sandbox is always deleted, even on errors
- **Detailed Logging**: All errors are logged with context
- **Statistics Tracking**: Track success/failure rates across models

### Logging

All operations are logged with timestamps:

- **Console Output**: Real-time progress and results
- **Log Files**: Persistent logs saved as `sandbox_YYYYMMDD_HHMMSS.log`

### Output Redirection

All sandbox commands properly redirect stdout and stderr:

- Commands use `2>&1` to capture all output
- Both stdout and stderr are logged via callbacks
- No output is lost during execution

### Statistics Summary

After pipeline completion, you'll see:

```
Pipeline Summary
============================================================
Models pulled: 3/3
Code generated: 3/3
Code executed: 3/3
Errors: 0
```

## GPU Support

The application supports GPU-enabled sandboxes for faster code generation. GPU availability depends on:

- Your Koyeb account tier (Pro or Scale plans typically have GPU access)
- Regional GPU availability
- Koyeb's GPU instance availability

### GPU Configuration

Configure GPU settings via command-line arguments:

```bash
# Request GPU and fail if unavailable (default)
python main.py

# Request GPU but allow CPU fallback
python main.py --no-require-gpu

# Use CPU-only sandbox
python main.py --no-gpu
```

**Behavior:**

- **Default** (`--no-gpu` not set, `--no-require-gpu` not set): Request GPU and fail if unavailable
- **`--no-require-gpu`**: Request GPU but continue with CPU if unavailable
- **`--no-gpu`**: Use CPU-only sandbox

You can also customize the GPU instance type:

```bash
python main.py --instance-type gpu-nvidia-rtx-4000-sff-ada --region fra
```

### Model Download Timeout

Models are automatically downloaded inside the sandbox. If a model fails to download:

- Check your internet connection
- Verify the model name is correct (e.g., "llama3.2" not "llama3")
- Larger models may take several minutes to download
- Progress is logged in real-time
- Pipeline continues with other models if one fails

### Sandbox Creation Timeout

Sandbox creation usually takes a few seconds. If it times out:

- Check your Koyeb account status
- Verify API token permissions
- Check if GPU instances are available in your region
- Try a different region or instance type

### Code Generation Failures

If code generation fails for a model:

- Check the log file for detailed error messages
- Verify the prompt is clear and specific
- Some models may not generate valid code for certain prompts
- Pipeline continues with other models

## Security Notes

- **Complete Isolation**: Both code generation and execution happen in isolated sandboxes
- **No Local AI Execution**: AI models never run on your local machine
- **Automatic Cleanup**: Sandboxes are automatically deleted after execution (guaranteed via try/finally)
- **Fresh Environments**: Each execution happens in a fresh, clean Ubuntu environment
- **No Persistent Storage**: No data persists between executions
- **Ephemeral Sandboxes**: Sandboxes cannot access your local system or other sandboxes
- **Secure Code Execution**: Generated code runs in isolated environment, cannot affect your system
- **API Token Security**: API token is only used for sandbox management, never exposed in sandbox

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [Koyeb Sandboxes Documentation](https://www.koyeb.com/docs/sandboxes/sandbox-quickstart)
- [Ollama Documentation](https://ollama.ai/docs)
- [Koyeb Python SDK](https://github.com/koyeb/koyeb-python-sdk)
