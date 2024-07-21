#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=01:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

CMD="multi-gpu-train"

. /etc/profile.d/modules.sh
module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

export NNODES=$NHOSTS
export NPERNODE=1
export NPERNODE=4
export NP=$(($NPERNODE * $NNODES))
export MASTER_ADDR=`head -n 1 $SGE_JOB_SPOOL_DIR/pe_hostfile | cut -d " " -f 1`
export MASTER_PORT=$((10000+ ($JOB_ID % 50000)))
echo NNODES=$NNODES
echo NPERNODE=$NPERNODE
echo NP=$NP
echo MASTERADDR=$MASTER_ADDR
echo MASTERPORT=$MASTER_PORT
echo CMD=$CMD

pwd
cd /gs/fs/tga-aklab/matsumoto/Main2
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2.2

mpirun -np $NP -npernode $NPERNODE python -m habitat_baselines.run --run-type train
