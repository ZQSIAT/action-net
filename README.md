# Pipeline
![avatar](/options/pipeline.png)
# Action-Net
Pytorch implementation of "A 3D Geometric Manifold Representation Method: Towards Good Practices forSkeleton Action Recognition".

## Introduction
Action-Net provides A 3D geometric manifold representation method, termed motion triangular planes of 3D joint (MTJ3D) is extracted from the raw skeleton data to capture the view invariant short-term motion cues. For long-term dependence, we construct 2D and 3D CNNs based backbone to aggregate the scattered frame-level features in space and time respectively. For purpose of updating the statistical weight of each motion triangular planes from a MTJ3D vector, the ablative experiments was carried out about exploring the structure of the Conv1. We also propose a multi-feature CNNs architecture to learn classification from complementary geometric feature manifolds.

## Requirements

1. Pytorch 0.4.1
2. CUDA 9.0 and cuDNN 7.0
3. openCV
## Usage
### Data Preparation
1. Please download the datasets i.e. UTD-MHAD, UTD-MVHAD, UWA3D, CAS-MHAD.

2. Configure dataset import file.
```
~/configs/datasets_config.json
~/configs/loader_config.json
~/configs/path_config.json
```
3. Generate test protocol
```
~/utils/generate_protocal_files.py
```
### Training
1. Configure arges options

```
~/options/args_option_defaults.json
```

2. Train Action-Net

```
main.py --code "***" --option "./options/args_option_defaults.json"
```

### Testing

1. Evaluate the model
```
main.py --evaluate "***_model_best.pth.tar"
```

## Citation
If you find Action-Net useful in your research, please kindly cite our paper:

```
@inproceedings{zhao2019view,
  title={View Invariant Human Action Recognition Using 3D Geometric Features},
  author={Zhao, Qingsong and Sun, Shijie and Ji, Xiaopeng and Wang, Lei and Cheng, Jun},
  booktitle={International Conference on Intelligent Robotics and Applications},
  pages={564--575},
  year={2019},
  organization={Springer}
}
```
