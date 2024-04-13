pwd
cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
#conda create -n habitat python=3.9 cmake=3.14.0
conda info -e
conda activate habitat
#python make_dataset.py
#python make_ci_and_similarity.py
python make_saliency_and_similarity.py
#python collect_environmental_picture.py
#CUDA_LAUNCH_BLOCKING=1 python PictureSelector/train_picture_selector.py
#python PictureSelector/train_picture_selector2.py
#python PictureSelector/check_selector_accuracy.py
#CUDA_LAUNCH_BLOCKING=1 python habitat_baselines/run.py --run-type eval
