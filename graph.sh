#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -j y
#$ -l h_rt=00:10:00
#$ -o output/o.$JOB_ID

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

pwd
cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

python ShowingGraph/ShowingRewardGraph.py
python ShowingGraph/ShowingEachRewardGraph.py
python ShowingGraph/ShowingActionGraph.py
python ShowingGraph/ShowingLossGraph.py
#python ShowingGraph/ShowingExpAreaGraph.py
#python ShowingGraph/ShowingExpAreaGraphCompare.py
python ShowingGraph/ShowingSimilarityGraph.py
#python ShowingGraph/ShowingSimilarityGraphCompare.py
#python ShowingGraph/ShowingSelectorAccuracyGraph.py
#python ShowingGraph/ShowingSelectorLossGraph.py
#python ShowingGraph/ShowingSaliencyAndSimilarityScatter.py