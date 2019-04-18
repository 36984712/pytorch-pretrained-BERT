#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="memory"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=127GB

module load pytorch/1.0.0-py37-cuda92

python3 ./examples/run_classifier_NER.py --data_dir ./BBN \
--bert_model bert-base-uncased \
--task_name bbn \
--output_dir ./output_model \
--do_train \
--do_lower_case \
--train_batch_size 10 \
--max_seq_length 16
