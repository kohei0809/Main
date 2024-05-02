#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -j y
#$ -l h_rt=1:00:00
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

git add --all
echo finish

#cd test_llava
#python test_llava.py
#python run.py --run-type random
#python run.py --run-type train
#python run.py --run-type eval
#python research_picture_value.py
#CUDA_LAUNCH_BLOCKING=1 python make_saliency_and_similarity.py
