#!/bin/bash

#SBATCH --job-name="sr-ulm_infer"
#SBATCH --time=18:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G
#SBATCH --qos=job_gpu_sznitman
#SBATCH --account=ws_00000
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_sr-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

python3 ./predict.py data_dir=/storage/workspaces/artorg_aimi/ws_00000/chris/Rat18_2D_InVivoBrain/ logging=pala_rat
