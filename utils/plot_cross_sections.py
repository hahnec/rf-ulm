from pathlib import Path
import imageio
import matplotlib.pyplot as plt

path = '../../Documents/insilico_paper/'
path = Path(path)

filenames = list(path.resolve().iterdir())
filenames = [filenames[4], filenames[2], filenames[0], filenames[3], filenames[5], filenames[1]]
labels = [x.name.split('_')[0] for x in filenames]
labels = ['Ground truth', 'Radial sym.', 'GULM', 'U-Net', 'mSPCN', 'SG-SPCN']
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
        axs2[i].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    axs2[i].set_ylim(0, line.max()*1.05)
    axs2[i].set_xlim(0, len(line))
    axs2[i].axis('off')
    axs2[i].annotate('', xy=(0.0, 1.0), xycoords='axes fraction', xytext=(0.0, 0.0), textcoords='axes fraction', arrowprops=dict(headwidth=8*4, headlength=8*4, color='k', width=2.5*4, shrink=1.1))
    axs2[i].annotate('', xy=(1.0, 0.0), xycoords='axes fraction', xytext=(0.0, 0.0), textcoords='axes fraction', arrowprops=dict(headwidth=8*4, headlength=8*4, color='k', width=2.5*4, shrink=1.1))
    #axs2[i].text(-.07, 0.45, 'Counts [a.u.]', fontsize=12, transform=axs2[i, 1].transAxes, rotation='vertical', va='center', ha='left')
    #axs2[i].text(0.4, -.125, 'Spatial axis [px]', fontsize=12, transform=axs2[i, 1].transAxes, rotation='horizontal', va='center', ha='left')

legend = axs2[0].legend(loc='upper right', fontsize=36, bbox_transform=axs2[1].transAxes)
legend.get_frame().set_alpha(1)  # Set the alpha value as needed
legend.set_bbox_to_anchor((1.05, 1.15))

# Save each subplot as an SVG file
fig2.set_facecolor('none')
for i in range(4):
    filename_cross = f'cross-section_{i+1}.pdf'
    extent = axs2[i].get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    fig2.savefig(filename_cross, bbox_inches=extent.expanded(1.1, 1.2), transparent=True)

plt.tight_layout()
fig2.savefig('cross-section.pdf')

plt.show()