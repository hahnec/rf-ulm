import wandb
import sys
from pathlib import Path
import json
import numpy as np
import shutil

api = wandb.Api()
runs = api.runs("SR-ULM-INFER")

# filter group runs
group_name = sys.argv[1] if len(sys.argv) > 1 else 'pala_insilico_array'
#runs = [run for run in runs if run.group == group_name]

# Sort the runs by creation time (most recent first)
sorted_runs = sorted(runs, key=lambda run: int(run.name.split('-')[-1]) if run.state == 'finished' else 0, reverse=True)

# retrieve a specific number of most recent runs
start_run = 1
num_recent_runs = 5
recent_runs = sorted_runs[start_run:start_run+num_recent_runs]

# artifact handling
load_artifact_opt = False
if load_artifact_opt and Path('./artifacts').exists(): shutil.rmtree('./artifacts')

ndigits = 3
metric_runs = []
toas, model_names = [], []
for metric_run in recent_runs:
    model_name = metric_run.summary['model_name'] if 'model_name' in metric_run.summary.keys() else None
    total_dist_mean = metric_run.summary['TotalRMSE'] * 10
    total_dist_std = metric_run.summary['TotalRMSEstd'] * 10
    total_time = metric_run.summary['FrameTime'] * 1e3
    total_params = metric_run.summary['TotalParameters']
    total_jaccard = metric_run.summary['TotalJaccard']
    ssim = metric_run.summary['SSIM'] * 1e2
    iqrf = metric_run.summary['Type'].lower().replace('iq', 'B-mode').replace('rf', 'RF-data')
    # replace None with '-'

    fmt = '.'+str(ndigits)+'f'
    model_name, total_dist_avg, total_dist_std, total_time, total_params, total_jaccard, ssim, iqrf = [format(round(el, ndigits), fmt) if isinstance(el, (float, int)) else el for el in [model_name, total_dist_mean, total_dist_std, total_time, total_params, total_jaccard, ssim, iqrf]]
    row_list = [str(model_name), str(iqrf), '$'+str(total_dist_avg)+' \pm '+str(total_dist_std)+'$', '$'+str(total_jaccard)+'$', '$'+str(ssim)+'$', '$'+str(total_params).split('.')[0]+'$', '$'+str(total_time)+'$']
    metric_runs.append(row_list)
    
    # download artifacts
    if load_artifact_opt:
        artifact_paths = []
        for artifact in metric_run.logged_artifacts():
            if artifact.type == "data":
                artifact_paths.append(Path('./artifacts') / model_name / artifact.name.split(':')[0])
                if load_artifact_opt: artifact.download(artifact_paths[-1])
        
        # load artifacts
        frame_idx = 2
        stack = []
        for name in ['toa', 'gt']:
            with open(artifact_paths[frame_idx] / (name+'.table.json'), 'r') as json_file:
                json_dict = json.load(json_file)
                stack.append(np.array(json_dict['data']))

        toa, gt = stack
        g=[np.load(f) for f in (artifact_paths[frame_idx]/'media'/'serialized_data').iterdir() if str(f).endswith('npz')][0]
        frame = g['Column0'].squeeze()

        # select channel from frame (PALA-only)
        if len(frame.shape) > 1:
            idx = 0
            frame = frame[idx, ...]
            toa = toa.squeeze()[idx]
            gt = gt.squeeze()[idx]

        toas.append(toa)
        model_names.append(model_name)

def write_metric_rows(f, metric_runs):

    # Iterate over the runs and extract metrics
    for k, row_list in enumerate(metric_runs):
        
        # replace model entry
        #row_list[0] = ['\multirow{2}{*}{mSPCN~\cite{liu2020deep}}', '', '\multirow{2}{*}{SG-SPCN (Ours)}', ''][k]
        row_list[0] = [
            '{mSPCN~\cite{liu2020deep}}', '{mSPCN~\cite{liu2020deep}}', 
            '{SG-SPCN (Ours)}', '{SG-SPCN (Ours)}',
            '',
                    ][k]

        # replace None entries
        row_list = [col.replace('None', 'n.a.') for col in row_list]

        # modify time
        row_list[-1] = '$\\text{T}_{\\text{DAS}}$ + ' + row_list[-1] if row_list[1] == 'B-mode' else row_list[-1]

        # Write a row for each metric in the LaTeX table
        f.write(" & ".join(str(metric) for metric in row_list))
        f.write(" \\\\\n")


# write table
with open("metrics_table.tex", "w") as f:
    # Write the LaTeX table header
    f.write("\\begin{tabularx}{\\textwidth}{ \n \
        >{\\raggedright\\arraybackslash}p{8em} %| \n \
        >{\\centering\\arraybackslash}p{5em} %| \n \
        >{\\raggedleft\\arraybackslash}p{8em} %| \n \
        >{\\raggedleft\\arraybackslash}p{8em} %| \n \
        >{\\raggedleft\\arraybackslash}p{8em} %| \n \
        >{\\raggedleft\\arraybackslash}p{8em} %| \n \
        >{\\raggedleft\\arraybackslash}p{9em} %| \n \
        }\n")
    f.write("\\toprule\n")
    f.write("Model & Input & {RMSE [$\lambda/10$]}~$\downarrow$ & {Jaccard~[\%]}~$\\uparrow$ & {SSIM~[\%]}~$\\uparrow$ & Weights [\#]~$\downarrow$ & Frame Time [ms]~$\downarrow$ \\\\\n")
    f.write("\\midrule\n")
    write_metric_rows(f, metric_runs)

    # Write the LaTeX table footer
    f.write("\\bottomrule\n")
    f.write("\\end{tabularx}")

# write table rows only
with open("metrics_rows.tex", "w") as f:
    write_metric_rows(f, metric_runs)
