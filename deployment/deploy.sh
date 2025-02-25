#!/bin/bash
set -e

# Configuration
DOCKER_REGISTRY="your-registry"  # Replace with your registry
IMAGE_NAME="llm-engine"
IMAGE_TAG=$(git rev-parse --short HEAD)  # Use git commit hash as tag

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest

# Run tests before deployment
echo "Running tests..."
docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} pytest

# Push to registry (uncomment when ready for production)
# echo "Pushing to registry..."
# docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
# docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
# docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
# docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest

# Deploy using docker-compose
echo "Deploying application..."
docker-compose -f docker-compose.yml up -d

# Wait for health check
echo "Waiting for application to be healthy..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health | grep -q '"status":"healthy"'; then
        echo "Application is healthy!"
        exit 0
    fi
    echo "Waiting for application to become healthy... (attempt $i/30)"
    sleep 5
done

echo "Error: Application failed to become healthy within timeout"
docker-compose logs
exit 1
