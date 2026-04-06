#!/bin/bash
# Final Fine-tuning: 9% (Weights) -> 5% (Target Error Rate)
# This script prevents macOS from sleeping (including lid close).
# Usage: ./train_finetune_05_final.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_05_FINAL="$SCRIPT_DIR/training_output_final_05.log"
PID_FILE="$SCRIPT_DIR/training_final_05.pid"
MODEL_09="$SCRIPT_DIR/outputs/attention_decoder_09err.pt"

# Prevent macOS from sleeping (including lid close)
echo "Disabling sleep (requires sudo)..."
sudo pmset -a disablesleep 1

source venv/bin/activate

(
  echo "[$(date)] Final Phase: Fine-tuning at 5% error, 120 epochs (warm-start from 9%)" >> "$LOG_05_FINAL"
  
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.05 \
    --n-train 5000 --n-val 1000 --n-test 1000 \
    --model-type attention \
    --hidden-dim 256 --num-layers 10 \
    --epochs 120 \
    --load-model "$MODEL_09" \
    --save-model \
    >> "$LOG_05_FINAL" 2>&1
    
  echo "[$(date)] Final Phase complete." >> "$LOG_05_FINAL"

  # Restore sleep when done
  sudo pmset -a disablesleep 0
  echo "[$(date)] All training done. Sleep restored." >> "$LOG_05_FINAL"

) &

echo $! > "$PID_FILE"
echo ""
echo "Final fine-tuning started — PID $(cat $PID_FILE)"
echo ""
echo "Monitor:  tail -f $LOG_05_FINAL"
echo ""
echo "Stop: kill \$(cat $PID_FILE)"
