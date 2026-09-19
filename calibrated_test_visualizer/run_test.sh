#!/bin/bash
# ==========================================================================
# Multi-Camera Triangulated Test — Deploy & Run
#
# Syncs project files to the server, ensures GPU dependencies are installed,
# runs the triangulated capture, copies outputs back, and opens HTML locally.
#
# Usage:
#   ./calibrated_test_visualizer/run_test.sh           # uses default height 188.5cm
#   ./calibrated_test_visualizer/run_test.sh 188.5      # explicit height in cm
#   ./calibrated_test_visualizer/run_test.sh 188.5 --calibration outputs/calibration.json
# ==========================================================================

set -e

# Configuration
LAPTOP_USER="ambaks"
LAPTOP_HOST="100.120.236.20"
SSH_TARGET="$LAPTOP_USER@$LAPTOP_HOST"
REMOTE_DIR="/home/$LAPTOP_USER/nowva"
LOCAL_OUTPUT_DIR="calibrated_test_visualizer/outputs"

ATHLETE_HEIGHT_CM="${1:-188.5}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  Multi-Camera Triangulated Test${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo "  Server:       $SSH_TARGET"
echo "  Remote dir:   $REMOTE_DIR"
echo "  Height:       ${ATHLETE_HEIGHT_CM} cm"
echo "  Extra args:   ${EXTRA_ARGS:-none}"
echo ""

# -----------------------------------------------------------------------
# Step 1: Check server connectivity
# -----------------------------------------------------------------------
echo -e "${BLUE}[1/7] Checking server connectivity...${NC}"
if ! ssh -o ConnectTimeout=5 "$SSH_TARGET" "echo OK" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Cannot reach $SSH_TARGET${NC}"
    echo "  Make sure the server is on and Tailscale is connected."
    exit 1
fi
echo -e "${GREEN}  Server reachable.${NC}"

# -----------------------------------------------------------------------
# Step 2: Ensure Python venv exists
# -----------------------------------------------------------------------
echo -e "${BLUE}[2/7] Checking Python environment...${NC}"
ssh "$SSH_TARGET" "
    if [ ! -d $REMOTE_DIR/venv ]; then
        echo 'Creating venv...'
        mkdir -p $REMOTE_DIR
        cd $REMOTE_DIR
        python3 -m venv venv
    fi
    echo 'venv OK'
"

# -----------------------------------------------------------------------
# Step 3: Ensure GPU dependencies
# -----------------------------------------------------------------------
echo -e "${BLUE}[3/7] Ensuring GPU dependencies (onnxruntime-gpu, opencv)...${NC}"
ssh "$SSH_TARGET" "
    cd $REMOTE_DIR && source venv/bin/activate

    # onnxruntime-gpu (replaces CPU version if present)
    if ! python3 -c 'import onnxruntime; assert \"CUDAExecutionProvider\" in onnxruntime.get_available_providers()' 2>/dev/null; then
        echo 'Installing onnxruntime-gpu...'
        pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
        pip install onnxruntime-gpu
    else
        echo 'onnxruntime-gpu OK'
    fi

    # opencv
    if ! python3 -c 'import cv2' 2>/dev/null; then
        echo 'Installing opencv...'
        pip install opencv-python-headless
    else
        echo 'opencv OK'
    fi

    # numpy (should already be there)
    pip install numpy --quiet 2>/dev/null
"

# -----------------------------------------------------------------------
# Step 4: Sync project files
# -----------------------------------------------------------------------
echo -e "${BLUE}[4/7] Syncing project files to server...${NC}"

ARCHIVE="/tmp/nowva-tri-test.tar.gz"

tar --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='venv' \
    --exclude='*.db' \
    --exclude='*.mp4' \
    --exclude='*.html' \
    --exclude='calibrated_test_visualizer/outputs/*' \
    -czf "$ARCHIVE" \
    src/ \
    scripts/demos/visualize_video_squats.py \
    scripts/tools/download_models.py \
    calibrated_test_visualizer/ \
    config/ \
    models/ 2>/dev/null || true

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "  Archive: $SIZE"

scp -q "$ARCHIVE" "$SSH_TARGET:~/nowva-tri-test.tar.gz"
ssh "$SSH_TARGET" "
    mkdir -p $REMOTE_DIR
    tar -xzf ~/nowva-tri-test.tar.gz -C $REMOTE_DIR
    rm ~/nowva-tri-test.tar.gz
"
echo -e "${GREEN}  Files synced.${NC}"

# -----------------------------------------------------------------------
# Step 5: Ensure RTMPose model is downloaded
# -----------------------------------------------------------------------
echo -e "${BLUE}[5/7] Checking RTMPose model...${NC}"
ssh "$SSH_TARGET" "
    cd $REMOTE_DIR && source venv/bin/activate
    MODEL_PATH='src/biomechanics/pose/models/rtmpose-m-halpe26-256x192.onnx'
    if [ ! -f \$MODEL_PATH ]; then
        echo 'Downloading RTMPose halpe26 model...'
        python3 scripts/tools/download_models.py --model rtmpose-m-halpe26
    else
        echo 'RTMPose model OK'
    fi
"

# -----------------------------------------------------------------------
# Step 6: Run the test
# -----------------------------------------------------------------------
echo -e "${BLUE}[6/7] Running triangulated capture on server...${NC}"
echo -e "${YELLOW}  (Follow instructions on the terminal — T-pose, then squat)${NC}"
echo ""

ssh -t "$SSH_TARGET" "
    cd $REMOTE_DIR && source venv/bin/activate
    export PYTHONPATH=$REMOTE_DIR/src:$REMOTE_DIR/scripts/demos
    python3 calibrated_test_visualizer/visualize_triangulated.py \
        --height $ATHLETE_HEIGHT_CM \
        --output-dir calibrated_test_visualizer/outputs \
        --no-open \
        $EXTRA_ARGS
"

# -----------------------------------------------------------------------
# Step 7: Copy outputs back and open HTML
# -----------------------------------------------------------------------
echo -e "${BLUE}[7/7] Copying outputs back...${NC}"

mkdir -p "$LOCAL_OUTPUT_DIR"

scp -q "$SSH_TARGET:$REMOTE_DIR/calibrated_test_visualizer/outputs/*" "$LOCAL_OUTPUT_DIR/" 2>/dev/null || {
    echo -e "${RED}No output files found on server.${NC}"
    exit 1
}

echo -e "${GREEN}  Outputs copied to $LOCAL_OUTPUT_DIR/${NC}"

# Open the latest HTML file
LATEST_HTML=$(ls -t "$LOCAL_OUTPUT_DIR"/*.html 2>/dev/null | head -1)
if [ -n "$LATEST_HTML" ]; then
    echo -e "${GREEN}  Opening: $LATEST_HTML${NC}"
    open "$LATEST_HTML"
else
    echo -e "${YELLOW}  No HTML file found in outputs.${NC}"
fi

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  Done! Outputs in $LOCAL_OUTPUT_DIR/${NC}"
echo -e "${GREEN}==========================================${NC}"
