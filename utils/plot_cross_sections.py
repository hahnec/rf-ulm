from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np

path = '../../Documents/insilico_paper/'
path = Path(path)

filenames = list(path.resolve().iterdir())
filenames = [filenames[4], filenames[2], filenames[0], filenames[3], filenames[5], filenames[1]]
labels = [x.name.split('_')[0] for x in filenames]
labels = ['Ground truth', 'RS', 'G-ULM', 'U-Net', 'mSPCN', 'SG-SPCN (Ours)']
colors = ['red', 'gray', 'violet', 'blue', 'orange', 'green']
linestyles = ['-', (0, (1, 1)), (5, (10, 3)), ':', '-.', '--']

crop_borders_1 = [240, 538, 1030, 238]
crop_borders_2 = [310, 45, 980, 750]
crop_borders_3 = [1240, 20, 0, 715]
crop_borders_4 = [820, 560, 200, 100]

all_borders = [crop_borders_1, crop_borders_2, crop_borders_3, crop_borders_4]
vertical_opt = [False, True, True, False]
vertical_line_scales = [.58,.78,.15,.34]

fig1_opt = False
if fig1_opt: fig1, axs1 = plt.subplots(len(all_borders), len(filenames))
fig2, axs2 = plt.subplots(len(all_borders), 1, figsize=(15, 27))
fig3, axs3 = plt.subplots(len(all_borders), 1, figsize=(15, 27))
for i, borders in enumerate(all_borders):
    for j, fname in enumerate(filenames):
        img = imageio.imread(fname)
        img_size = img.shape
        crop = img[borders[3]:img_size[0]-borders[1], borders[0]:img_size[1]-borders[2]]
        crop_size = crop.shape
        if fig1_opt:axs1[i, j].imshow(crop)
        vertical_idx = int(crop_size[not vertical_opt[i]]*vertical_line_scales[i])
        line_startx = crop.shape[1]//2
        line_stopx = -int(line_startx/2.25)
        line_startx += 2
        if vertical_opt[i]:
            line = crop[line_startx:line_stopx, vertical_idx:vertical_idx+1].mean(-1).squeeze(1)
            crop[line_startx:line_stopx, vertical_idx:vertical_idx+1] = img.max()    # mark selected line
        else:
            line = crop[vertical_idx:vertical_idx+1, line_startx:line_stopx].mean(-1).squeeze(0)
            crop[vertical_idx:vertical_idx+1, line_startx:line_stopx] = img.max()    # mark selected line
        axs2[i].plot(line, label=str(labels[j]), linestyle=linestyles[j], color=colors[j], lw=6)
        if j == 0:
            axs3[i].imshow(crop)
            imageio.imsave(f'gt_crop_{i+1}.png', crop)
        if fig1_opt:axs1[i, j].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        axs3[i].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        #axs2[i].tick_params(top=False, bottom=True, left=True, right=False, labelleft=False, labelbottom=False)

        axs2[i].set_ylim(0, line.max()*1.05)
        axs2[i].set_xlim(0, len(line))
        #axs2[i].axis('off')
        arrow_settings = dict(headwidth=8 * 4, headlength=8 * 4, color='k', width=2.5 * 4, shrink=1.1, linewidth=2.0)
        axs2[i].annotate('', xy=(0.0, 1.0), xycoords='axes fraction', xytext=(0.0, 0.0), textcoords='axes fraction', arrowprops=arrow_settings)
        axs2[i].annotate('', xy=(1.0, 0.0), xycoords='axes fraction', xytext=(0.0, 0.0), textcoords='axes fraction', arrowprops=arrow_settings)

        # Set fewer ticks with larger font size
        if len(line) > 32:
            desired_xticks = np.linspace(0, len(line)//4*4, len(line)//8+1).astype('int') 
        elif len(line) > 8:
            desired_xticks = np.linspace(0, len(line)//4*4, len(line)//4+1).astype('int') 
        else:
            desired_xticks = np.linspace(0, len(line)-1, len(line)-1)
        desired_yticks = np.arange(0, 81, 40).astype('int')
        axs2[i].set_xticks(desired_xticks)
        axs2[i].set_yticks(desired_yticks)

        # Customize ticks along the arrowed axes
        axs2[i].tick_params(axis='both', which='both', direction='out', length=18, width=2.5 * 4)
        axs2[i].tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelbottom=True)
        #axs2[i].set_xticks(axs2[i].get_xticks()[:-1])  # Leave out the last tick on the x-axis
        #axs2[i].set_yticks(axs2[i].get_yticks()[:-1])  # Leave out the last tick on the y-axis
        axs2[i].set_xticklabels([int(el) for el in axs2[i].get_xticks()], fontsize=24*2)
        axs2[i].set_yticklabels([int(el) for el in axs2[i].get_yticks()], fontsize=24*2)

        if False:
            major_ticks = range(0, len(line), 1) # Set major ticks every 5th tick
            axs2[i].set_xticks(major_ticks)
            minor_ticks = [tick for tick in line if tick not in major_ticks]
            axs2[i].minorticks_on()
            axs2[i].set_xticks(minor_ticks, minor=True)

        axs2[i].spines['top'].set_visible(False)
        axs2[i].spines['right'].set_visible(False)

legend = axs2[0].legend(loc='upper right', fontsize=36, bbox_transform=axs2[1].transAxes)
legend.get_frame().set_alpha(1)  # Set the alpha value as needed
legend.set_bbox_to_anchor((1.05, 1.15))

# Save each subplot as an SVG file
fig2.set_facecolor('none')
for i in range(4):
    filename_cross = f'cross-section_{i+1}.pdf'
    extent = axs2[i].get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    # Pad the saved area by 20% in the x-direction and 30% in the y-direction
    #extent = extent.expanded(1.2, 1.3)
    #extent = (extent[0] - 0.2, extent[1], extent[2] - 0.3, extent[3])
    y0scale = [0.0425, 0.06, 0.0975, 0.2725][i]
    y1scale = [-0.019, -0.001, -0.001, -0.001][i]
    extended_extent = extent.from_extents(extent.x0-0.6*extent.x0, extent.y0-extent.y0*y0scale, extent.x1+0.026*extent.x1, extent.y1-extent.y1*y1scale)

    fig2.savefig(filename_cross, bbox_inches=extended_extent, transparent=True)

plt.tight_layout()
fig2.savefig('cross-section.pdf')

#plt.show()