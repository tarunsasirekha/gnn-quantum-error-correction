#!/bin/bash
# Fine-tune on 7% error rate, warm-starting from saved checkpoint
# Chain: 7% (150 epochs) → 9% (150 epochs)
# Usage: ./train_07.sh (needs sudo password once)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_7="$SCRIPT_DIR/training_output_finetune_07.log"
LOG_9="$SCRIPT_DIR/training_output_finetune_09v2.log"
PID_FILE="$SCRIPT_DIR/training.pid"
CHECKPOINT="$SCRIPT_DIR/outputs/attention_decoder.pt"

echo "Disabling sleep (requires sudo)..."
sudo pmset -a disablesleep 1

source venv/bin/activate

(
  # ── 7% error rate, 150 epochs, warm-start ──
  echo "[$(date)] Fine-tune: 7% error, 150 epochs (warm-start from checkpoint)" >> "$LOG_7"
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.07 \
    --n-train 6000 --n-val 1000 --n-test 1000 \
    --model-type attention \
    --hidden-dim 256 --num-layers 10 \
    --epochs 150 \
    --load-model "$CHECKPOINT" \
    --save-model \
    >> "$LOG_7" 2>&1
  echo "[$(date)] 7% run complete." >> "$LOG_7"

  # ── 9% error rate, 150 epochs, warm-start from 7% checkpoint ──
  echo "[$(date)] Fine-tune: 9% error, 150 epochs (warm-start from 7% checkpoint)" >> "$LOG_9"
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.09 \
    --n-train 8000 --n-val 1000 --n-test 1000 \
    --model-type attention \
    --hidden-dim 256 --num-layers 10 \
    --epochs 150 \
    --load-model "$SCRIPT_DIR/outputs/attention_decoder.pt" \
    --save-model \
    >> "$LOG_9" 2>&1
  echo "[$(date)] 9% run complete." >> "$LOG_9"

  sudo pmset -a disablesleep 0
  echo "[$(date)] All done. Sleep restored." >> "$LOG_9"

) &

echo $! > "$PID_FILE"
echo ""
echo "Training started — PID $(cat $PID_FILE)"
echo ""
echo "7% run:  tail -f $LOG_7"
echo "9% run:  tail -f $LOG_9 (starts automatically after 7%)"
echo "Stop:    kill \$(cat $PID_FILE)"
