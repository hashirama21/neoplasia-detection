#!/usr/bin/env bash
# do_save.sh — Export Docker image as .tar.gz for Grand Challenge upload
# Usage: ./do_save.sh [output_filename]

set -euo pipefail

IMAGE_NAME="rare26"
OUTPUT_FILE="${1:-rare26_submission.tar.gz}"

echo "======================================================"
echo " RARE26 Docker Export"
echo "======================================================"

# Verify image exists
if ! docker image inspect "$IMAGE_NAME:latest" &>/dev/null; then
  echo "ERROR: Image $IMAGE_NAME:latest not found. Run ./do_test_run.sh first."
  exit 1
fi

echo "Exporting $IMAGE_NAME:latest → $OUTPUT_FILE ..."
echo "This may take several minutes (image size: $(docker image inspect $IMAGE_NAME:latest --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo 'unknown'))..."

docker save "$IMAGE_NAME:latest" | gzip > "$OUTPUT_FILE"

SIZE=$(du -sh "$OUTPUT_FILE" | cut -f1)
echo ""
echo "Export complete: $OUTPUT_FILE ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload $OUTPUT_FILE to Grand Challenge → Submit → Open Development Phase"
echo "  2. After upload, verify container becomes 'Active' (20-60 minutes)"
echo "  3. Run test on a sample image in the Grand Challenge UI"
echo "======================================================"
