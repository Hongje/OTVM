## One-Trimap Video Matting (ECCV 2022)
### Hongje Seong, Seoung Wug Oh, Brian Price, Euntai Kim, Joon-Young Lee

This is the implementation of OTVM (ECCV 2022).  
This code is based on TCVOM (ACM MM 2021): [[link](https://github.com/yunkezhang/TCVOM)]  
To see the demo video, please click the below image.  
[![One-Trimap Video Matting (ECCV 2022)](https://img.youtube.com/vi/qkda4fHSyQE/0.jpg)](https://youtu.be/qkda4fHSyQE "One-Trimap Video Matting (ECCV 2022)")

## Environments
- Ubuntu 18.04
- python 3.8
- pytorch 1.8.2
- CUDA 10.2

### Environment setting
```bash
conda create -n otvm python=3.8
conda activate otvm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-contrib-python scikit-image scipy tqdm imgaug yacs albumentations
```

## Dataset
To train OTVM, you need to prepare [DIM](https://sites.google.com/view/deepimagematting) and [VideoMatting108](https://github.com/yunkezhang/TCVOM) datasets
```
PATH/TO/DATASET
├── Combined_Dataset
│   ├── Adobe Deep Image Mattng Dataset License Agreement.pdf
│   ├── README.txt
│   ├── Test_set
│   │   ├── Adobe-licensed images
│   │   └── ...
│   └── Training_set
│   │   ├── Adobe-licensed images
│   │   └── ...
└── VideoMatting108
    ├── BG_done2
    │   ├── airport
    │   └── ...
    ├── FG_done
    │   ├── animal_still
    │   └── ...
    ├── flow_png_val
    │   ├── animal_still
    │   └── ...
    ├── frame_corr.json
    ├── train_videos_subset.txt
    ├── train_videos.txt
    ├── val_videos_subset.txt
    └── val_videos.txt

```

## Training
### Download pre-trained weights
Note: Initial weights of the trimap propagation and alpha prediction networks were taken from [STM](https://github.com/seoungwugoh/STM) and [FBA](https://github.com/MarcoForte/FBA_Matting), respectively.
```bash
mkdir weights
wget -O weights/STM_weights.pth "https://www.dropbox.com/s/mtfxdr93xc3q55i/STM_weights.pth?dl=1"
wget -O weights/FBA.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EZjHx4oY0-RIkEanfwZTW4oBPov7q6KdfybGriHnm51feQ?download=1"
wget -O weights/s1_OTVM_trimap.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EVjt5DzKqp5FoHOQpvZ3ydIB9WKIk180-BeRns2TpKhALA?download=1"
wget -O weights/s1_OTVM_alpha.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EZwARPMb9TxCpLK3krxAmygBgOEVn6_mjA_Fiqa8OVrhIQ?download=1"
wget -O weights/s2_OTVM_alpha.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EUJX1ZILpCFMldsd013eWlkB_erJC53FfOATUx7XWWtKxw?download=1"
wget -O weights/s3_OTVM.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EeTwD4A-ElNCvZjhlSALqoABrg239xmuym0z28UOKDHj5A?download=1"
wget -O weights/s4_OTVM.pth "https://yonsei-my.sharepoint.com/:u:/g/personal/hjseong_o365_yonsei_ac_kr/EdJy_a2QfGpPhCBbXUf_Bg4Bd2DFGgYRZEr2ifaQy76qiw?download=1"
```
### Change DATASET.PATH in config.py
```bash
vim config.py

# Change below path
_C.DATASET.PATH = 'PATH/TO/DATASET'
```

### Stage-wise Training
```bash
# options: scripts/train_XXX.sh [GPUs]
bash scripts/train_s1_trimap.sh 0,1,2,3
bash scripts/train_s1_alpha.sh 0,1,2,3
bash scripts/train_s2_alpha.sh 0,1,2,3
bash scripts/train_s3.sh 0,1,2,3
bash scripts/train_s4.sh 0,1,2,3
```

## Inference
```bash
# options: scripts/eval_s4.sh [GPU]
bash scripts/eval_s4.sh 0
```

## Bibtex
```
@inproceedings{seong2022one,
  title={One-Trimap Video Matting},
  author={Seong, Hongje and Oh, Seoung Wug and Price, Brian and Kim, Euntai and Lee, Joon-Young},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```


## Terms of Use
This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)

[![Creative Commons License](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
