#!/bin/bash

#SBATCH --job-name="sr-ulm_train"
#SBATCH --time=18:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --account=ws_00000
#SBATCH --qos=job_gpu_sznitman
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --array=1-5%5

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_sr-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

param_store=~/20_sr-ulm/bash_scripts/array_pala_params.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
type=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')

python3 ./train.py model=${model} model_file=${model_file} nms_threshold=${threshold} input_type=${type} pow_law_opt=False clutter_db=-50 sequences=[16,17,18,19,20] data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/PALA_data_InSilicoFlow/
