## Uncertainty quantification using Bayesian neural networks in classification: Application to ischemic stroke lesion segmentation (version 0.0.2)

Keras implementation of the paper "Uncertainty quantification using Bayesian neural networks in classification". This github is under construction at June 2018.


I strongly recommend to see the good implementation using the DRIVE dataset by Walter de Back. [[notebook](https://gitlab.com/wdeback/dl-keras-tutorial/blob/master/notebooks/3-cnn-segment-retina-uncertainty.ipynb)].


## Directories

```bash
.
├── ischemic
│   ├── input
│   └── src
│       ├── configs (empty)
│       ├── data.py
│       ├── models.py
│       ├── settings.py
│       ├── train.py
│       ├── utils.py
│ 		└── weights (empty)
├── README.md
└── retina
    ├── fig
    ├── input
    │   └── (train/test datasets form DRIVE)
    ├── model.py
    ├── UQ_DRIVE_stochastic_sample_2000.ipynb
    ├── UQ_DRIVE_stochastic_sample_200.ipynb
    ├── utils.py
    └── weights (empty)

```

## Ischemic stroke lesion segmentation

## Dataset

Ischemic stroke lesion segmentation 2015 dataset (XXX)

## Code

- `data.py` loads '.nii' image files efficiently 
- `train.py` main train file
- `models.py` contains the scripts for the Bayesian dropout and network architectures (multiscale 3D U-Net).

## Usage 

```bash
python train.py --cnf c_spes_kendall # for the Kendall and Gal implementation
python train.py --cnf c_spes_proposed # for the Kwon et al. implementation
```

## Retina vessel segmentation

## Dataset

DRIVE dataset (XXX)

## Code

- `models.py` contains the scripts for the Bayesian dropout and network architectures (2D U-Net).
- `utils.py` contains functions for the prediction and the preprocessing procedures.

## Usage 

Implementations are detailed at `UQ_DRIVE_stochastic_sample_2000.ipynb` and `UQ_DRIVE_stochastic_sample_200.ipynb`.




