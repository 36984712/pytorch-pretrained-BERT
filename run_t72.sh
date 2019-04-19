#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="bbn t72 l32 b16"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=127GB

module load pytorch/1.0.0-py37-cuda92

python3 ./run_bbn.py --data_dir ./BBN 
--bert_model bert-base-uncased \
--task_name bbn \
--output_dir ./output_model_l32_b16_t32 \
--do_train \
--do_lower_case \
--train_batch_size 32 \
--max_seq_length 128 \
--num_train_epochs 50 \
--do_eval