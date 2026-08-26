#!/bin/bash
#SBATCH -A <truba-account>
#SBATCH -p akya-cuda
#SBATCH -J mobilenet_gpu
#SBATCH -o mobilenet_gpu.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# === Ortam ayarları ===
module purge
module load apps/truba-ai/gpu-2024.0

echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info
echo "==========================="
echo "🚀 Mobilenet GPU Test"
echo "HOSTNAME       : $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "==========================="

# === Python GPU kontrolü ===
python - <<'EOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
EOF

# === Mobilenet testini çalıştır ===
python /arf/scratch/<truba-user>/mobileNetGPU/mobilenet_gpu_test.py

exit
