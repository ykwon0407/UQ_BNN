### UQ for Ischemic Stroke Lesion Segmentation 2015 (ISLES 2015)

``The codes assume the 'channel-first' order.``

## Getting started (Need to download ISLES datasets at the ISLES website)

```bash
python train.py --cnf c_spes_kendall # for the Kendall and Gal implementation
python train.py --cnf c_spes_proposed # for the Kwon et al. implementation
```

## Files

- `data.py` loads '.nii' image files efficiently 
- `train.py` main train file
- `models.py` contains the scripts for the Bayesian dropout and network architectures (multiscale 3D U-Net).

## References

- [ISLES website](http://www.isles-challenge.org/)
- Oskar Maier et al. ISLES 2015 - A public evaluation benchmark for ischemic stroke lesion segmentation from multispectral MRI, Medical Image Analysis, Available online 21 July 2016, ISSN 1361-8415, http://dx.doi.org/10.1016/j.media.2016.07.009. 
