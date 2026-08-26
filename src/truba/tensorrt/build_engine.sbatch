#!/bin/bash
#SBATCH -J build_trt_engine
#SBATCH -p akya-cuda
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

# --- Environment setup ---
module purge
module load apps/truba-ai/gpu-2024.0

# --- Safety ---
set -e
mkdir -p logs
mkdir -p engines

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"

# --- Run TensorRT engine build ---
python3 build_engine.py \
  --onnx mobilenetv2.onnx \
  --engine engines/mobilenetv2_bs1.trt \
  --batch 1 \
  --fp16

echo "End time    : $(date)"
echo "✅ TensorRT engine build finished"
