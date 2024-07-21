#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=00:20:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

#module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

#export MAGNUM_LOG=verbose 
#export MAGNUM_GPU_VALIDATION=ON

cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#python test_llava/test_llava.py
python create_viewer_video.py