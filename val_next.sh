#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -j y
#$ -l h_rt=10:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

echo "LLaVA-NEXT"

cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2


pip install transformers==4.45.2
pip install accelerate==0.26.0

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CUDA_LAUNCH_BLOCKING=1 python run.py --run-type random
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type novelty
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type smooth-coverage
#CUDA_LAUNCH_BLOCKING=1 python run.py --run-type eval5 --area-reward-type curiosity

echo "LLaVA-NEXT"