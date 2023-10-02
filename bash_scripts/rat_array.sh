#!/bin/bash

#SBATCH --job-name="rf-ulm_bench"
#SBATCH --time=42:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=job_gpu_sznitman
#SBATCH --account=ws_00000
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --array=1-1%3

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_rf-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

param_store=~/20_rf-ulm/bash_scripts/array_rat_params_12x.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
type=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')
skip_bmode=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $6}')

python3 ./predict.py model=${model} model_file=${model_file} nms_threshold=${threshold} input_type=${type} skip_bmode=${skip_bmode} clutter_db=Null dither=False batch_size=1 pow_law_opt=False track=all_in data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/Rat18_2D_InVivoBrain/ logging=pala_rat synth_gt=False
