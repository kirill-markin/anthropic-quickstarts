#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY environment variable is not set"
  echo "Please set it: export ANTHROPIC_API_KEY=your_key"
  exit 1
fi

# Current directory must be computer-use-demo
if [ ! -f "Dockerfile" ]; then
  echo "Error: Dockerfile not found. Make sure you're in the computer-use-demo directory"
  exit 1
fi

# Stop and remove container if already running
if docker ps -a | grep -q "claude-local"; then
  echo "Stopping existing container..."
  docker stop claude-local 2>/dev/null || true
  docker rm claude-local 2>/dev/null || true
fi

# Build local Docker image from current code
echo "Building Docker image from local code..."
if [ "$1" == "--no-cache" ]; then
  echo "Building without using Docker cache..."
  docker build --no-cache -t claude-computer-use:local .
else
  docker build -t claude-computer-use:local .
fi

# Set log levels, respecting environment variable if set
STREAMLIT_LOGLEVEL=${STREAMLIT_LOGLEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# If STREAMLIT_LOGLEVEL is debug, set LOG_LEVEL to DEBUG too
if [ "$STREAMLIT_LOGLEVEL" = "debug" ]; then
  LOG_LEVEL=DEBUG
  echo "Setting LOG_LEVEL to DEBUG because STREAMLIT_LOGLEVEL is debug"
fi

echo "Starting container with log settings:"
echo "  - STREAMLIT_LOGLEVEL=$STREAMLIT_LOGLEVEL"
echo "  - LOG_LEVEL=$LOG_LEVEL"
echo "  - Logs will be saved to computer-use-demo/logs/ directory"

# Run container with mounted local code
echo "Starting container with local code..."
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -e ENABLE_THINKING=${ENABLE_THINKING:-true} \
    -e STREAMLIT_LOGLEVEL=$STREAMLIT_LOGLEVEL \
    -e LOG_LEVEL=$LOG_LEVEL \
    -e PYTHONUNBUFFERED=1 \
    -v $(pwd)/computer_use_demo:/home/computeruse/computer_use_demo/ \
    -v $(pwd)/logs:/home/computeruse/logs \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -e WIDTH=${WIDTH:-1920} \
    -e HEIGHT=${HEIGHT:-1080} \
    --name claude-local \
    -it claude-computer-use:local

echo "Container stopped"
