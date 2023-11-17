#!/bin/bash

#SBATCH --job-name="rf-ulm_infer"
#SBATCH --time=18:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=job_gpu_sznitman
#SBATCH --account=ws_00000
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:rtx3090:1

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA/11.8.0

source ~/20_rf-ulm/venv/bin/activate

python3 -c "import torch; print(torch.cuda.is_available())"

python3 ./predict.py clutter_db=Null wv_idcs=[0,1,2] eps=.5 logging=pala_rat
