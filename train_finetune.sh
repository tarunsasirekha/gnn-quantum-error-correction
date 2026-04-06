#!/bin/bash
# Fine-tunes from saved checkpoints — no training from scratch.
# Chain: 5% (100 more epochs) → 9% (150 more epochs)
# Usage: ./train_finetune.sh (run from terminal, needs sudo password once)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_5="$SCRIPT_DIR/training_output_finetune_05.log"
LOG_9="$SCRIPT_DIR/training_output_finetune_09.log"
PID_FILE="$SCRIPT_DIR/training.pid"
CHECKPOINT="$SCRIPT_DIR/outputs/attention_decoder.pt"

echo "Disabling sleep (requires sudo)..."
sudo pmset -a disablesleep 1

source venv/bin/activate

(
  # ── Fine-tune 1: load 9% checkpoint → 100 more epochs at 5% ──
  echo "[$(date)] Fine-tune 1: 5% error, 100 epochs (warm-start)" >> "$LOG_5"
  caffeinate -dims python main.py \
    --code-type toric \
    --code-size 4 \
    --error-rate 0.05 \
    --n-train 4000 --n-val 800 --n-test 800 \
    --model-type attention \
    --hidden-dim 256 --num-layers 10 \
    --epochs 100 \
    --load-model "$CHECKPOINT" \
    --save-model \
    >> "$LOG_5" 2>&1
  echo "[$(date)] Fine-tune 1 complete." >> "$LOG_5"

  # ── Fine-tune 2: load updated checkpoint → 150 more epochs at 9% ──
  echo "[$(date)] Fine-tune 2: 9% error, 150 epochs (warm-start)" >> "$LOG_9"
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
  echo "[$(date)] Fine-tune 2 complete." >> "$LOG_9"

  sudo pmset -a disablesleep 0
  echo "[$(date)] All done. Sleep restored." >> "$LOG_9"

) &

echo $! > "$PID_FILE"
echo ""
echo "Fine-tuning started — PID $(cat $PID_FILE)"
echo ""
echo "Fine-tune 1 (5% error, 100 epochs):"
echo "  Monitor: tail -f $LOG_5"
echo ""
echo "Fine-tune 2 (9% error, 150 epochs) — starts automatically after:"
echo "  Monitor: tail -f $LOG_9"
echo ""
echo "Stop: kill \$(cat $PID_FILE)"
