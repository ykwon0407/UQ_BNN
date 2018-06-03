## Uncertainty quantification using Bayesian neural networks in classification: Application to ischemic stroke lesion segmentation (version 0.0.1)

Pytorch implementation of the paper "Uncertainty quantification using Bayesian neural networks in classification". This github is under construction at April 2018.

## Scripts

```bash
.
├── input
├   ├── (train/test datasets...)
├── README.md
└── src
    ├── configs
    ├── data.py
    ├── loggings (empty)
    ├── models.py
    ├── settings.py
    ├── train.py
    ├── utils.py
    └── weights (empty)
```

- `data.py` loads '.nii' image files efficiently 
- `train.py` main train file
- `models.py` contains the scripts for the Bayesian dropout and network architectures (multiscale 3D U-Net).

## Usage

```bash
python train.py --cnf c_spes_kendall
```

## Creators

**Yongchan Kwon**
