#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A oceylan
#SBATCH -J barbun_max
#SBATCH -o barbun_max.out
#SBATCH --gres=gpu:2             # Her 20 CPU için 1 GPU → 40 CPU = 2 GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00
#SBATCH --mem=64G

# === Modül yükleme ===
module purge
module load comp/oneapi/2023
module load lib/cuda/12.1              # 12.2 mevcut değilse 12.1 tercih edilir
module load lang/python/3.10-venv      # 3.11 genelde mevcut değil

# === Python sanal ortamını aktive et ===
source ~/venvs/tfenv/bin/activate

# === GPU tanıma kontrolü ===
python - <<'EOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
EOF

# === Asıl MobileNet GPU testi ===
python /arf/scratch/oceylan/mobileNetGPU/mobilenet_gpu_test.py
