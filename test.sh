#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -j y
#$ -l h_rt=01:00:00
#$ -o output/o.$JOB_ID
#$ -p -4

. /etc/profile.d/modules.sh

module load openmpi/5.0.2-gcc
module load cuda/12.1.0 

cd /gs/fs/tga-aklab/matsumoto/Main
. /home/7/ur02047/anaconda3/etc/profile.d/conda.sh
conda activate habitat2

#pip list
#pip install git+https://github.com/huggingface/transformers
#pip install --upgrade protobuf
#pip install -U accelerate
#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

pip install accelerate==0.21.0
pip install huggingface-hub==0.22.2
pip install transformers==4.37.2

#pip install transformers==4.45.2
#pip install accelerate==0.26.0


python test_llava/test_llava.py
#python test_llava/test_llava2.py
#python test_llava/test_llava_next.py
#python human_metrics.py
#cd SentenceBert_FineTuning
#python finetuning_sbert.py
#python finetuning_sbert2.py
#python finetuning_sbert3.py
#python finetuning_sbert4.py
#python finetuning_sbert5.py
#python ShowingFinetuninng.py