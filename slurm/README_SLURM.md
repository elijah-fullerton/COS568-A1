# Slurm guide for COS568-A1 Task 1 on neuronic

Neuronic node specs: 2×26-core CPUs, 512 GB RAM, 8× NVIDIA L40, 3.5 TB local SSD. These scripts target the Task 1 (hyper-parameter tuning) grid at 10% sparsity.

## Files
- `scripts/run_hparam.py` — drives the Task 1 grid (architectures × pruners).
- `slurm/run_part1.sbatch` — submits the full grid on a single GPU.

## What the grid covers (Task 1)
- Architectures: (cifar10, lottery, resnet20), (mnist, default, fc)
- Pruners: rand, mag, snip, grasp, synflow
- Compression exponent: 1 → sparsity = 10%
- Pre-epochs: 200 for mag, 0 for others
- Results: `Results/data/singleshot/part1_*` plus `part1_summary.json`

## How to submit
```bash
cd /u/ef0952/COS568/torch_demo/COS568-A1
mkdir -p slurm_logs
sbatch slurm/run_part1.sbatch
```

## Monitoring
- Queue: `squeue -u $USER`
- Logs: `tail -f slurm_logs/part1_<jobid>.out`

## Resource choices (tunable)
- `--gres=gpu:1` (one L40)
- `--cpus-per-task=8` (good for 4–6 dataloader workers)
- `--mem=32G` (can lower to 16 G after confirming headroom)
- `--time=12:00:00` (trim once you know runtime)
- `--partition=gpu` (use actual GPU partition name if different)
- Add `--account/--qos` if your site requires them.

## Customizing
- Change epochs/batch sizes by editing the arguments in `run_part1.sbatch`.
- Use `--dry-run` in `scripts/run_hparam.py` to print commands only.
- You can add git commit/push steps after the run if desired.

## Quick rationale
- Single GPU per job: minimizes contention; each neuronic node has 8× L40.
- 8 CPUs: ample for the provided batch sizes and 4 workers.
- 32 G RAM: generous for CIFAR/MNIST; reduce after a test run.
