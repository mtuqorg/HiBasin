# HiBasin

The Hierarchical BAyesian Source INversion (HiBASIN) is a Python package to perform seismic moment tensor (MT), single force (SF), or joint MT and SF inversion within a hierarchical Bayesian framework incoporating uncertainty estimates for data noise and theory error. Hibsasin is based on [MTUQ](https://github.com/mtuqorg/mtuq) and [emcee](https://github.com/dfm/emcee).  

## Installation

1. Requirements:
    * [MTUQ](https://github.com/mtuqorg/mtuq) ([https://github.com/mtuqorg/mtuq](https://github.com/mtuqorg/mtuq))
    * [emcee](https://github.com/dfm/emcee) ([https://github.com/dfm/emcee](https://github.com/dfm/emcee))
    * [pyrocko](https://git.pyrocko.org/pyrocko/pyrocko)
    * [basemap](https://github.com/matplotlib/basemap)
    * [corner](https://corner.readthedocs.io/en/latest/)

 
2. Install HiBasin:
```shell
git clone git@github.com:mtuqorg/HiBasin.git
cd HiBasin
pip install -e .
```

## Documentation
Read the MTUQ documentation for [Acquiring seismic data](https://mtuqorg.github.io/mtuq/user_guide/02.html), [Acquiring Green's functions](https://mtuqorg.github.io/mtuq/user_guide/03.html), and [Data  processing](https://mtuqorg.github.io/mtuq/user_guide/04.html). Note that, at least one-hour long pre-event ambient noise should be included in the downloaded seismic data. A cutting noise windown will be used to estimate the noise. 

## Examples
1. Full moment tensor inversion using HiBasin for a tectonic earthquake.

|  uncorrelated noise treatment     |     correlated noise treatment        |
|:---------------------------------:|:-------------------------------------:|
| [Script](./examples/EMCEE.FullMomentTensor.AK20090407.py), [Figure](docs/images/FMT_Bay_uncorr_AK2009.pdf)  | [Script](./examples/EMCEE.FullMomentTensor.AK20090407_sharedCd.py), [Figure](docs/images/FMT_Bay_corr_AK2009.pdf)      |
   * Check the [here](docs/images/FMT_Bay_tau_k_AK2009.pdf) for inverted time shifts and noise parameters.
   * Check [here](docs/images/AK2009_covariance_matrix_sharedCd_2att.png) for estimated covariance matrix for correlated noise.


2. Full moment tensor inversion using HiBasin for six DPRK explosions in 2006–2017 by considering data noise.

|       2006       |       2009       |       2013       |       2016a      |       2016b      |       2017       |
|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| [Script](./examples/EMCEE.FullMomentTensor.DPRK2006.py), [Figure](docs/images/FMT_Bay_uncorr_2006.pdf) | [Script](./examples/EMCEE.FullMomentTensor.DPRK2009.py), [Figure](docs/images/FMT_Bay_uncorr_2009.pdf) | [Script](./examples/EMCEE.FullMomentTensor.DPRK2013.py), [Figure](docs/images/FMT_Bay_uncorr_2013.pdf) | [Script](./examples/EMCEE.FullMomentTensor.DPRK2016a.py), [Figure](docs/images/FMT_Bay_uncorr_2016a.pdf) | [Script](./examples/EMCEE.FullMomentTensor.DPRK2016b.py), [Figure](docs/images/FMT_Bay_uncorr_2016b.pdf) | [Script](./examples/EMCEE.FullMomentTensor.DPRK2017.py), [Figure](docs/images/FMT_Bay_uncorr_2017.pdf) |
   * Check [here](docs/images/FMT_Bay_uncorr_tau_k_2017.pdf) for inverted time shift and noise parameters.

3. Tutorial for 1D model uncertainty treatment.
   * Perturb the reference 1D model
   * Compute the ensemble of Green's functions
   * Estimate the covariance matrix by giving a reference moment tensor solution
  
  Check the [script](./util/greens_ensemble.py) for details and the [script](./examples/prepare_greens_ensemble_DPRK2013) and [figure](docs/images/Cm_matrix_mt_itr0.png) for an example. 

## Citation:
1. Hu, J., T.-S., Phạm, & H., Tkalčić, (2023). Seismic moment tensor inversion with theory errors from 2-D Earth structure: implications for the 2009–2017 DPRK nuclear blasts. Geophysical Journal International, 235(3), 2035–2054. 
2. Mustać, M. & H., Tkalčić. (2016). Point source moment tensor inversion through a Bayesian hierarchical model. Geophysical Journal International, 204 (1), 311-323.
3. Phạm, T.-S. and H., Tkalčić. (2021). Toward improving point‐source moment‐tensor inference by incorporating 1d earth model's uncertainty: Implications for the long valley caldera earthquakes. Journal of Geophysical Research: Solid Earth 126 (11), e2021JB022477.
