#!/bin/bash
# COS568-A1 Task 1 grid on neuronic (SR670 V2: 2x26c, 512G RAM, 8x L40)
# Resources below are ample for CIFAR/MNIST single-GPU runs.
# Adjust partition/account/qos if your site requires it.
#SBATCH --job-name=cos568-part1
#SBATCH --partition=all          # neuronic partitions: all (7d), lowprio (7d), interactive (2h)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ef0952@cs.princeton.edu
#SBATCH --output=slurm_logs/part1_%j.out
#SBATCH --error=slurm_logs/part1_%j.err

echo "Node(s): $SLURM_NODELIST"

# Activate your environment (adjust to your venv/conda path)
source /u/ef0952/COS568/torch_demo/venv/bin/activate

cd /u/ef0952/COS568/torch_demo/COS568-A1

# Run Task 1 grid (10% sparsity). Magnitude uses pre-epochs=200; others 0.
python scripts/run_hparam.py \
  --repo-root . \
  --results-root Results/data/singleshot \
  --compression 1 \
  --post-epochs 10 \
  --pre-epochs-mag 200 \
  --pre-epochs-default 0 \
  --train-batch-size 256 \
  --test-batch-size 256 \
  --prune-batch-size 256 \
  --workers 4 \
  --seed 1
