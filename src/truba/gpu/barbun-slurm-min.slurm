#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A oceylan
#SBATCH -J barbun_min
#SBATCH -o barbun_min.out
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# === Modül yükleme ===
module purge
module load comp/oneapi/2023
module load lib/cuda/12.1              # 12.2 mevcut değilse 12.1 kullan
module load lang/python/3.10-venv      # 3.11 yoksa 3.10 versiyonu çalışır

# === Sanal ortamı aktive et ===
source ~/venvs/tfenv/bin/activate

# === TensorFlow GPU kontrolü ===
python - <<'EOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
EOF

# === Ana script ===
python /arf/scratch/oceylan/mobileNetGPU/mobilenet_gpu_test.py
