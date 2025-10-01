#!/bin/bash

# Vision-CLS Docker Runner
# Provides interactive bash access and Jupyter Lab server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="vision-cls-container"
IMAGE_NAME="vision-cls:latest"
JUPYTER_PORT=8888
HOST_JUPYTER_PORT=8888

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vision-CLS Docker Environment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker first"
    exit 1
fi

# Check for NVIDIA GPU support
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    GPU_FLAGS="--gpus all"
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected, running in CPU mode${NC}"
fi

# Stop existing container if running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} > /dev/null 2>&1
fi

# Remove existing container if exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm ${CONTAINER_NAME} > /dev/null 2>&1
fi

# Check if image exists, if not build it
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
    echo -e "${YELLOW}Image not found. Building ${IMAGE_NAME}...${NC}"
    docker build -t ${IMAGE_NAME} .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Image built successfully${NC}"
    else
        echo -e "${RED}Error: Failed to build image${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Using existing image: ${IMAGE_NAME}${NC}"
fi

echo ""
echo -e "${BLUE}Starting container...${NC}"
echo ""

# Run container with Jupyter Lab
docker run -d \
    ${GPU_FLAGS} \
    --name ${CONTAINER_NAME} \
    -p ${HOST_JUPYTER_PORT}:${JUPYTER_PORT} \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/log:/app/log" \
    -v "$(pwd)/notebooks:/app/notebooks" \
    -v "$(pwd)/src:/app/src" \
    --restart unless-stopped \
    ${IMAGE_NAME}

# Wait for container to start
sleep 3

if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo -e "${GREEN}✓ Container started successfully${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Access Information${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${GREEN}Jupyter Lab:${NC} http://localhost:${HOST_JUPYTER_PORT}"
    echo -e "${YELLOW}Note: No authentication required${NC}"
    echo ""
    echo -e "${BLUE}To access container bash:${NC}"
    echo -e "  ${YELLOW}docker exec -it ${CONTAINER_NAME} /bin/bash${NC}"
    echo ""
    echo -e "${BLUE}To run training:${NC}"
    echo -e "  ${YELLOW}docker exec -it ${CONTAINER_NAME} python3 src/main.py --dataset_name your_dataset${NC}"
    echo ""
    echo -e "${BLUE}To view logs:${NC}"
    echo -e "  ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
    echo ""
    echo -e "${BLUE}To stop:${NC}"
    echo -e "  ${YELLOW}./stop.sh${NC}"
    echo ""
    echo -e "${GREEN}Opening browser in 3 seconds...${NC}"
    sleep 3

    # Try to open browser
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:${HOST_JUPYTER_PORT}" &> /dev/null &
    elif command -v open &> /dev/null; then
        open "http://localhost:${HOST_JUPYTER_PORT}" &> /dev/null &
    else
        echo -e "${YELLOW}Please open http://localhost:${HOST_JUPYTER_PORT} in your browser${NC}"
    fi
else
    echo -e "${RED}Error: Container failed to start${NC}"
    echo "Check logs with: docker logs ${CONTAINER_NAME}"
    exit 1
fi
