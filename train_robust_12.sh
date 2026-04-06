#!/bin/bash
# Robustness Training: 12% (Pre-train) -> 9% (Fine-tune)
# This script prevents macOS from sleeping (including lid close).
# Usage: ./train_robust_12.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_12="$SCRIPT_DIR/training_output_12err.log"
LOG_9_FT="$SCRIPT_DIR/training_output_finetune_09v2.log"
PID_FILE="$SCRIPT_DIR/training_12_9.pid"
MODEL_12="$SCRIPT_DIR/outputs/attention_decoder_12err.pt"

# Prevent macOS from sleeping (including lid close)
echo "Disabling sleep (requires sudo)..."
sudo pmset -a disablesleep 1

source venv/bin/activate

(
  # ── Phase 1: Pre-train at 12% error rate ──
  # This provides the "robust" weights
  echo "[$(date)] Phase 1: Pre-training at 12% error, 200 epochs" >> "$LOG_12"
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.12 \
    --n-train 10000 \
    --n-val 1000 \
    --n-test 1000 \
    --model-type attention \
    --hidden-dim 256 \
    --num-layers 10 \
    --epochs 200 \
    --save-model \
    --output-dir "$SCRIPT_DIR/outputs" \
    >> "$LOG_12" 2>&1
  
  # Move the 12% model to a specific name so it's not overwritten by Phase 2
  mv "$SCRIPT_DIR/outputs/attention_decoder.pt" "$MODEL_12"
  echo "[$(date)] Phase 1 complete. Model saved to $MODEL_12" >> "$LOG_12"

  # ── Phase 2: Fine-tune at 9% error rate ──
  # Loading the robust 12% weights
  echo "[$(date)] Phase 2: Fine-tuning at 9% error, 150 epochs (warm-start from 12%)" >> "$LOG_9_FT"
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.09 \
    --n-train 8000 \
    --n-val 1000 \
    --n-test 1000 \
    --model-type attention \
    --hidden-dim 256 \
    --num-layers 10 \
    --epochs 150 \
    --load-model "$MODEL_12" \
    --save-model \
    >> "$LOG_9_FT" 2>&1
  echo "[$(date)] Phase 2 complete." >> "$LOG_9_FT"

  # Restore sleep when both are done
  sudo pmset -a disablesleep 0
  echo "[$(date)] All training done. Sleep restored." >> "$LOG_9_FT"

) &

echo $! > "$PID_FILE"
echo ""
echo "Robustness training started — PID $(cat $PID_FILE)"
echo ""
echo "Phase 1 (12% error, 200 epochs):"
echo "  Monitor:  tail -f $LOG_12"
echo ""
echo "Phase 2 (9% error, 150 epochs fine-tune) — starts automatically after Phase 1:"
echo "  Monitor:  tail -f $LOG_9_FT"
echo ""
echo "Stop both: kill \$(cat $PID_FILE)"
