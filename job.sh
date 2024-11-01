#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=01:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

#apptainer exec --nv ./../ubuntu_latest.sif /gs/fs/tga-aklab/matsumoto/Main/exec.sh
#apptainer shell --nv -B /gs/fs/tga-aklab/matsumoto/Main:/gs/fs/tga-aklab/matsumoto/Main ./../_ubuntu_latest.sif 
#apptainer shell --nv -B /gs/fs/tga-aklab/matsumoto/ -B /home/7/ur02047/ ./../_ubuntu_latest.sif 

pwd

#cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train5 --area-reward-type coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train5 --area-reward-type novelty
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train5 --area-reward-type smooth-coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train5 --area-reward-type curiosity
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train5 --area-reward-type reconstruction
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train3
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train2
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train --area-reward-type coverage
#python run.py --run-type eval2
#python research_picture_value.py

#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type collect --area-reward-type coverage
