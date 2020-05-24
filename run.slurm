#!/bin/bash
#SBATCH -J nmt
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00
#SBATCH -o test.out

bash train.sh