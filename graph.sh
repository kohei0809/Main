#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o output/o.$JOB_ID

source ~/anaconda3/etc/profile.d/conda.sh

module load cuda
module load gcc/8.3.0-cuda
module load singularity
module load nccl
module load cudnn
module load openmpi/3.1.4-opa10.10

singularity exec -f --nv --bind /gs/hs0/tga-aklab/matsumoto://root/work ./../nvidia_cudagl.img ./work/Main/exec_graph.sh