#!/bin/bash

#SBATCH --job-name="rf-ulm_bench"
#SBATCH --time=14:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --array=1-5%5

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_rf-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

param_store=~/20_rf-ulm/bash_scripts/array_pala_params.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
type=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')
skip_bmode=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $6}')

python3 ./predict.py model=${model} model_file=${model_file} nms_threshold=${threshold} input_type=${type} batch_size=1 clutter_db=-50 skip_bmode=${skip_bmode} sequences=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] upscale_factor=8 nms_size=11 eps=0.5 wv_idcs=[0,1,2] track=all_in pow_law_opt=False dither=True synth_gt=False data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/PALA_data_InSilicoFlow/ logging=pala_insilico_array
