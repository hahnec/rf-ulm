#!/bin/bash

#SBATCH --job-name="sr-ulm_bench"
#SBATCH --time=18:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-3%1

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_sr-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

param_store=~/20_sr-ulm/bash_scripts/array_pala_params.txt

model=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
model_file=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
threshold=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')

python3 ./predict.py model=${model} model_file=${model_file} th=${threshold} batch_size=1 data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/PALA_data_InSilicoFlow/ logging=pala_array
