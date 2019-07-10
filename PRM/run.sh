#!/bin/bash
#SBATCH --job-name=once
#SBATCH --partition=gpu  
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --ntasks-per-node 1
#SBATCH --output=./once.out
#SBATCH --error=./once.err
#SBATCH --gres=gpu:1
    srun python once.py --cuda True