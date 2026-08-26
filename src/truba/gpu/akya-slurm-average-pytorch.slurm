#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A oceylan
#SBATCH -J akya_average_pytorch
#SBATCH -o akya_average_pytorch.out
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mem=32G

# === Ortam Ayarları ===
module purge
module load apps/truba-ai/gpu-2024.0   # TensorFlow ve PyTorch için hazır ortam (CUDA + Python + TF/PyTorch)
echo "✅ Loaded TRUBA AI GPU environment"

# === Çalıştırma Dizini ===
cd /arf/scratch/oceylan/mobileNetGPU

# === TensorFlow Mobilenet Test ===
python mobilenet_gpu_pytorch.py