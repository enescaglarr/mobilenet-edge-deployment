#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A <truba-account>
#SBATCH -J akya_min
#SBATCH -o akya_min.out
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# === Ortam Ayarları ===
module purge
module load apps/truba-ai/gpu-2024.0   # TensorFlow ve PyTorch için hazır ortam (CUDA + Python + TF/PyTorch)
echo "✅ Loaded TRUBA AI GPU environment"

# === Çalıştırma Dizini ===
cd /arf/scratch/<truba-user>/mobileNetGPU

# === TensorFlow Mobilenet Test ===
python mobilenet_gpu_test.py