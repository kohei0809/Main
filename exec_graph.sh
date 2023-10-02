pwd
cd work/Main
. ~/anaconda3/etc/profile.d/conda.sh
conda activate habitat
python ShowingGraph/ShowingRewardGraph.py
python ShowingGraph/ShowingActionGraph.py
python ShowingGraph/ShowingCIGraph.py
python ShowingGraph/ShowingLossGraph.py
python ShowingGraph/ShowingExpAreaGraph.py
python ShowingGraph/ShowingTakePicture.py
#python ShowingGraph/ShowingTakePictureCompare.py