pwd
cd work/Main
. ~/anaconda3/etc/profile.d/conda.sh
conda activate habitat
python ShowingGraph/ShowingRewardGraph.py
python ShowingGraph/ShowingEachRewardGraph.py
#python ShowingGraph/ShowingEachRewardGraphCompare.py
#python ShowingGraph/ShowingRewardGraphCompare.py
python ShowingGraph/ShowingActionGraph.py
python ShowingGraph/ShowingCIGraph.py
python ShowingGraph/ShowingCIGraphCompare.py
python ShowingGraph/ShowingLossGraph.py
python ShowingGraph/ShowingExpAreaGraph.py
python ShowingGraph/ShowingObjectNumGraph.py
#python ShowingGraph/ShowingCIMap.py
