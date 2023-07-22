#!/bin/bash

#SBATCH --job-name="sr-ulm_infer"
#SBATCH --time=06:00:00

#SBATCH --mail-user=christopher.hahne@unibe.ch
#SBATCH --mail-type=none
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=job_gpu
#SBATCH --account=ws_00000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:1

module load Python/3.8.6-GCCcore-10.2.0
#module load CUDA/11.3.0-GCC-10.2.0
#module load cuDNN/8.2.0.53-CUDA-11.3.0
#module load Workspace

python -m venv venv

source ~/20_ulm_unet/venv/bin/activate

python -m pip install -r requirements.txt

git clone --recurse-submodules git@github.com:hahnec/simple_tracker simple_tracker_repo
#python3 -m pip install -r simple_tracker_repo/requirements.txt
ln -sf ./simple_tracker_repo/simple_tracker simple_tracker

python -c "import torch; print(torch.cuda.is_available())"

python ./predict.py