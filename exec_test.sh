pwd
cd work/Main
. ~/anaconda3/etc/profile.d/conda.sh
conda activate habitat
#python make_dataset.py
#python make_ci_map.py
CUDA_LAUNCH_BLOCKING=1 python habitat_baselines/run.py 
