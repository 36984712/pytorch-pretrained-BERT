#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="BBN_ALL"
#SBATCH --time=64:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=127GB


module load python/3.6.1
module load cuda/8.0.61
module load cudnn/v7.0.3

python ./run_classifier_NER.py --data_dir ./BBN \
--bert_model bert-base-uncased \
--task_name bbn \
--output_dir ../output \
--do_train \
--do_lower_case \
