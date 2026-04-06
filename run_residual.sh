#!/bin/bash
echo "Waiting for attention run (PID 3475) to finish..."
wait 3475
echo "Attention run complete. Starting residual run at $(date)"
cd /Users/tarunsasirekha/Downloads/gnn-quantum-error-correction-main
nohup venv/bin/python main.py \
  --code-type toric \
  --code-size 6 \
  --model-type residual \
  --epochs 4000 \
  --n-train 3000 \
  --num-layers 6 \
  --save-model \
  --plot-results \
  --save-plots > residual_training_output.log 2>&1
echo "Residual run complete at $(date)"
