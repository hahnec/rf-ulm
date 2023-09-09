from pathlib import Path
import imageio
import matplotlib.pyplot as plt

path = '../../Documents/insilico_paper/'
path = Path(path)

filenames = list(path.resolve().iterdir())
filenames = [filenames[4], filenames[2], filenames[0], filenames[3], filenames[5], filenames[1]]
labels = [x.name.split('_')[0] for x in filenames]
colors = ['red', 'gray', 'violet', 'blue', 'orange', 'green']
linestyles = ['-', (0, (1, 1)), (0, (3, 5, 1, 5, 1, 5)), ':', '-.', '--']

crop_borders_1 = [240, 538, 1030, 238]
crop_borders_2 = [310, 45, 980, 750]
crop_borders_3 = [1240, 20, 0, 720]
crop_borders_4 = [820, 560, 200, 100]

all_borders = [crop_borders_1, crop_borders_2, crop_borders_3, crop_borders_4]
vertical_opt = [False, True, True, False]
vertical_line_scales = [.58,.78,.5,.35]

fig1, axs1 = plt.subplots(len(all_borders), len(filenames))
fig2, axs2 = plt.subplots(len(all_borders), 2)
for i, borders in enumerate(all_borders):
    for j, fname in enumerate(filenames):
        img = imageio.imread(fname)
        img_size = img.shape
        crop = img[borders[3]:img_size[0]-borders[1], borders[0]:img_size[1]-borders[2]]
        crop_size = crop.shape
        axs1[i, j].imshow(crop)
        vertical_idx = int(crop_size[not vertical_opt[i]]*vertical_line_scales[i])
        if vertical_opt[i]:
            line = crop[:, vertical_idx:vertical_idx+1].mean(-1).squeeze(1)
            crop[:, vertical_idx:vertical_idx+1] = img.max()    # mark selected line
        else:
            line = crop[vertical_idx:vertical_idx+1, :].mean(-1).squeeze(0)
            crop[vertical_idx:vertical_idx+1, :] = img.max()    # mark selected line
        axs2[i, 1].plot(line, label=str(labels[j]), linestyle=linestyles[j], color=colors[j])
        if j == 0:
            axs2[i, 0].imshow(crop)
        axs1[i, j].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        axs2[i, 0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        axs2[i, 1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
plt.legend()
plt.show()