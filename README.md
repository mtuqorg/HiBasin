# HiBasin

The Hierarchical BAyesian Source INversion (HiBASIN) is a Python package to perform seismic moment tensor (MT), single force (SF), or joint MT and SF inversion within a hierarchical Bayesian framework incoporating uncertainty estimates for data noise and theory error. Hibsasin is based on [MTUQ](https://github.com/mtuqorg/mtuq) and [emcee](https://github.com/dfm/emcee).  

## Installation

1. Requirements:
    * [MTUQ](https://github.com/mtuqorg/mtuq) ([https://github.com/mtuqorg/mtuq](https://github.com/mtuqorg/mtuq))
    * [emcee](https://github.com/dfm/emcee) ([https://github.com/dfm/emcee](https://github.com/dfm/emcee))
    * [corner](https://corner.readthedocs.io/en/latest/)
    * [pyrocko](https://git.pyrocko.org/pyrocko/pyrocko)
 
2. Install HiBasin:
```shell
git clone git@github.com:mtuqorg/HiBasin.git
cd HiBasin
pip install -e .
```

## Documentation
Read the MTUQ documentation for [Acquiring seismic data](https://mtuqorg.github.io/mtuq/user_guide/02.html), [Acquiring Green's functions](https://mtuqorg.github.io/mtuq/user_guide/03.html), and [Data  processing](https://mtuqorg.github.io/mtuq/user_guide/04.html).

## Examples
Inversion result of the 2017 DPRK nuclear test obtained by HiBasin.
   * [Script](./examples/EMCEE.FullMomentTensor.DPRK2017.py) 
   * [Figure](docs/images/FMT_Bay_uncorr.pdf)

## Citation:
Hu, J., Phạm, T.-S., & Tkalčić, H. (2023). Seismic moment tensor inversion with theory errors from 2-D Earth structure: implications for the 2009–2017 DPRK nuclear blasts. Geophysical Journal International, 235(3), 2035–2054. https://doi.org/10.1093/gji/ggad348
