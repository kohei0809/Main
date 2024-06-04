#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=95:40:00
#$ -o output/o.$JOB_ID

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

CUDA_LAUNCH_BLOCKING=1 python run.py --run-type train2
#python run.py --run-type eval2
#python research_picture_value.py
