import sys
import wandb
from pathlib import Path
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import imageio

normalize = lambda x: np.round((x-x.min())/(x.max()-x.min())*255).astype(np.uint8)

api = wandb.Api()
runs = api.runs("SR-ULM-INFER")

insilico = True
if insilico:
    run_idx = 4-2
    frame_idx = 10+2
    group_name = 'pala_insilico_array'
else:
    run_idx = 0
    frame_idx = 3 #4 #8 #4
    group_name = 'pala_rat'

# filter group runs
group_name = sys.argv[1] if len(sys.argv) > 1 else group_name
runs = [run for run in runs if run.group == group_name]

# Sort the runs by creation time (most recent first)
sorted_runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]) if run.state == 'Running' else 0, reverse=True)

# retrieve a specific number of most recent runs
metric_run = sorted_runs[run_idx]

imgs = []
for key in ["magnitude_img", "localization_img"]:
    s = metric_run.history(keys=[key], pandas=False)
    relative_img_path = s[frame_idx][key]['path']
    img_url = 'https://api.wandb.ai/files/hahnec/SR-ULM-INFER/' + metric_run.url.split('runs/')[-1] + '/' + relative_img_path
    response = requests.get(img_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Load the image from the response content using Pillow
        pil_object = Image.open(BytesIO(response.content))

        # Convert the image to a NumPy array
        img = np.array(pil_object)
        imgs.append(normalize(img))
        imageio.imsave(key.split('_')[0]+'.png', imgs[-1])
