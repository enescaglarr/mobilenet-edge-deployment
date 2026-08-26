#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A <truba-account>
#SBATCH -J akya_max
#SBATCH -o akya_max.out
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --mem=64G


# === Ortam Ayarları ===
module purge
module load apps/truba-ai/gpu-2024.0   # TensorFlow ve PyTorch için hazır ortam (CUDA + Python + TF/PyTorch)
echo "✅ Loaded TRUBA AI GPU environment"

# === Çalıştırma Dizini ===
cd /arf/scratch/<truba-user>/mobileNetGPU

# === TensorFlow Mobilenet Test ===
python mobilenet_gpu_test.py