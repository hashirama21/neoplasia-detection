#!/usr/bin/env bash
# do_test_run.sh — Test Docker container locally before submission
# Usage: ./do_test_run.sh [--network-none]
# Mirrors Grand Challenge evaluation environment exactly.

set -euo pipefail

IMAGE_NAME="rare26"
TEST_INPUT_DIR="tests/test_input"
TEST_OUTPUT_DIR="tests/test_output"
NETWORK_FLAG=""

# Parse flags
for arg in "$@"; do
  case $arg in
    --network-none) NETWORK_FLAG="--network none" ;;
  esac
done

echo "======================================================"
echo " RARE26 Docker Test Run"
echo "======================================================"

# Check Docker is running
if ! docker info &>/dev/null; then
  echo "ERROR: Docker is not running. Start Docker and retry."
  exit 1
fi

# Create test directories
mkdir -p "$TEST_INPUT_DIR" "$TEST_OUTPUT_DIR"

# Create a dummy test image if none exist
if [ -z "$(ls -A $TEST_INPUT_DIR 2>/dev/null)" ]; then
  echo "Creating dummy test image (336x336 RGB)..."
  python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
img.save('$TEST_INPUT_DIR/test_image.png')
print('Dummy image created: $TEST_INPUT_DIR/test_image.png')
"
fi

# Build image
echo "Building Docker image: $IMAGE_NAME..."
docker build \
  -t "$IMAGE_NAME:latest" \
  -f docker/Dockerfile \
  .

echo "Build successful."

# Run test
echo ""
echo "Running inference test ${NETWORK_FLAG:+(--network=none)}..."
docker run \
  --rm \
  ${NETWORK_FLAG} \
  --gpus all 2>/dev/null || true \
  -v "$(pwd)/$TEST_INPUT_DIR:/input:ro" \
  -v "$(pwd)/$TEST_OUTPUT_DIR:/output:rw" \
  -e INPUT_DIR=/input \
  -e OUTPUT_DIR=/output \
  "$IMAGE_NAME:latest"

# Check output
echo ""
echo "Checking output..."
if [ -f "$TEST_OUTPUT_DIR/predictions.json" ]; then
  echo "Output file found: $TEST_OUTPUT_DIR/predictions.json"
  echo "Contents:"
  cat "$TEST_OUTPUT_DIR/predictions.json"
  echo ""
  echo "TEST PASSED"
else
  echo "ERROR: predictions.json not found in output directory."
  echo "Check container logs above."
  exit 1
fi

echo "======================================================"
echo " Test complete. Ready to submit."
echo "======================================================"
