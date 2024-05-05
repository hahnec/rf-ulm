## RF-ULM: Radio-Frequency Ultrasound Localization Microscopy

[![arXiv paper link](https://img.shields.io/badge/paper-arXiv:2306.08281-red)](https://arxiv.org/pdf/2310.01545.pdf)

### Overview
<div style="background-color: white;">
<img src="https://github.com/hahnec/rf-ulm/blob/master/docs/rf-ulm_concept.svg" width="500" scale="100%">
</div>
NMS: Non-Maximum-Suppression
<br>
Map: Geometric point transformation from RF to B-mode coordinate space
<br>
<br>

### SG-SPCN Architecture
<div style="background-color: white;">
<img src="https://github.com/hahnec/rf-ulm/blob/master/docs/rf-ulm_arch.svg" width="780" scale="100%">
</div>
<br>

### Demos
#### 1. ULM Animation Demo
<video src="https://github.com/hahnec/rf-ulm/assets/33809838/e37aee11-c07f-4d9b-8672-5a9b466edd26" controls autoplay loop muted>
    Link: https://github.com/hahnec/rf-ulm/assets/33809838/e37aee11-c07f-4d9b-8672-5a9b466edd26
</video>
<b>Note</b>: The video starts in slow motion and then exponentially increases the frame rate for better visualization.
<br>

#### 2. Prediction Frames Demo
<video src="https://github.com/hahnec/rf-ulm/assets/33809838/4f4002bb-01e1-405f-aa56-e3c6b7a3b654" controls autoplay loop muted>
    Link: https://github.com/hahnec/rf-ulm/assets/33809838/4f4002bb-01e1-405f-aa56-e3c6b7a3b654
</video>

<b>Note</b>: Colors represent localizations from each plane wave emission angle.

### Datasets

*In vivo* (inference): https://doi.org/10.5281/zenodo.7883227

*In silico* (training+inference): https://doi.org/10.5281/zenodo.4343435
<br>
<br>

### Short presentation at IUS 2023

[<img src="https://img.youtube.com/vi/eJJXnXay-fU/hqdefault.jpg" width="480" height="360"
/>](https://www.youtube.com/embed/eJJXnXay-fU)

### Installation

It is recommended to use a UNIX-based system for development. For installation, run (or work along) the following bash script:

```
> bash install.sh
```

Note that the dataloader module is missing in this repository. My implementation is a hacky version of the work found at https://github.com/AChavignon/PALA, which was used as a reference in this project. When using data other than mentioned here, one would need to start writing this part from scratch. The simpletracker repository has not been used in the TMI publication and can be ignored.

### Citation

If you use this project for your work, please cite:

```
@inproceedings{hahne:2023:learning,
    author = {Christopher Hahne and Georges Chabouh and Olivier Couture and Raphael Sznitman},
    title = {Learning Super-Resolution Ultrasound Localization Microscopy from Radio-Frequency Data},
    booktitle= {2023 IEEE International Ultrasonics Symposium (IUS)},
    address={},
    month={Sep},
    year={2023},
    pages={1-4},
}
```

<!--
```
@misc{rfulm:2023,
      title={RF-ULM: Deep Learning for Radio-Frequency Ultrasound Localization Microscopy}, 
      author={Christopher Hahne and Georges Chabouh and Arthur Chavignon and Olivier Couture and Raphael Sznitman},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
-->

### Acknowledgment

This research is funded by the Hasler Foundation under project number 22027.
