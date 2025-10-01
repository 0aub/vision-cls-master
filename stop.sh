#!/bin/bash

# Vision-CLS Docker Stopper
# Cleanly stops and removes the container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="vision-cls-container"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vision-CLS Docker Cleanup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if container is running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo -e "${YELLOW}Stopping container: ${CONTAINER_NAME}...${NC}"
    docker stop ${CONTAINER_NAME}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Container stopped successfully${NC}"
    else
        echo -e "${RED}Error: Failed to stop container${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Container is not running${NC}"
fi

# Check if container exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo -e "${YELLOW}Removing container: ${CONTAINER_NAME}...${NC}"
    docker rm ${CONTAINER_NAME}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Container removed successfully${NC}"
    else
        echo -e "${RED}Error: Failed to remove container${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Container does not exist${NC}"
fi

echo ""
echo -e "${GREEN}✓ Cleanup completed${NC}"
echo ""
echo -e "${BLUE}Your data and logs are preserved in:${NC}"
echo -e "  - ${YELLOW}./data/${NC}"
echo -e "  - ${YELLOW}./log/${NC}"
echo -e "  - ${YELLOW}./notebooks/${NC}"
echo ""
echo -e "${BLUE}To restart the container, run:${NC}"
echo -e "  ${YELLOW}./run.sh${NC}"
echo ""
