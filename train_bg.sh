#!/bin/bash
# Runs training detached from terminal, preventing macOS sleep (including lid close)
# Automatically chains into the next experiment after the first finishes.
# Usage: ./train_bg.sh
# Monitor: tail -f training_output_AB.log
# Stop:    kill $(cat training.pid)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_1="$SCRIPT_DIR/training_output_AB.log"
LOG_2="$SCRIPT_DIR/training_output_09err.log"
PID_FILE="$SCRIPT_DIR/training.pid"

# Prevent macOS from sleeping (including lid close)
echo "Disabling sleep (requires sudo)..."
sudo pmset -a disablesleep 1

source venv/bin/activate

(
  # ── Experiment 1: 5% error rate, 200 epochs (already running — skip if done) ──
  if [ ! -f "$LOG_1" ] || ! grep -q "EXPERIMENT COMPLETE" "$LOG_1"; then
    echo "[$(date)] Starting Experiment 1: attention, L=4, 5% error, 200 epochs" >> "$LOG_1"
    caffeinate -dims python main.py \
      --code-type toric \
      --code-size 4 \
      --error-rate 0.05 \
      --n-train 4000 \
      --n-val 800 \
      --n-test 800 \
      --model-type attention \
      --hidden-dim 256 \
      --num-layers 10 \
      --epochs 200 \
      --save-model \
      >> "$LOG_1" 2>&1
    echo "[$(date)] Experiment 1 complete." >> "$LOG_1"
  else
    echo "[$(date)] Experiment 1 already complete, skipping." >> "$LOG_2"
  fi

  # ── Experiment 2: 9% error rate, 250 epochs ──
  echo "[$(date)] Starting Experiment 2: attention, L=4, 9% error, 250 epochs" >> "$LOG_2"
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
    --epochs 250 \
    --save-model \
    >> "$LOG_2" 2>&1
  echo "[$(date)] Experiment 2 complete." >> "$LOG_2"

  # Restore sleep when both are done
  sudo pmset -a disablesleep 0
  echo "[$(date)] All training done. Sleep restored." >> "$LOG_2"

) &

echo $! > "$PID_FILE"
echo ""
echo "Training started — PID $(cat $PID_FILE)"
echo ""
echo "Experiment 1 (5% error, 200 epochs):"
echo "  Monitor:  tail -f $LOG_1"
echo ""
echo "Experiment 2 (9% error, 250 epochs) — starts automatically after Exp 1:"
echo "  Monitor:  tail -f $LOG_2"
echo ""
echo "Stop both: kill \$(cat $PID_FILE)"
