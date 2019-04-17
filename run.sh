#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="BBN_16_shot"
#SBATCH --time=64:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=127GB

module load pytorch/1.0.0-py37-cuda92

python3 ./run_classifier_NER.py --data_dir ./BBN \
--bert_model bert-base-uncased \
--task_name bbn \
--output_dir ./output_model \
--do_train \
--do_lower_case \
--num_train_epochs 1000
