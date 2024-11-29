#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

pwd

#cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#pip install accelerate==0.21.0
#pip install huggingface-hub==0.22.2
#pip install transformers==4.37.2

CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval --area-reward-type coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval4
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type novelty
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type smooth-coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type curiosity