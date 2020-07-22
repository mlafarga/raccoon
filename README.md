# raccoon

Radial velocities and Activity indicators from Cross-COrrelatiON with masks

`raccoon` is an open source python package the implements the cross-correlation function (CCF) method.
It is able to
- build weighted binary masks from a stellar spectrum template
- compute the CCF of stellar spectra with a mask
- derive radial velocities (RVs) and activity indicators from the CCF

The methods implemented in this code are explained in this article: https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..36L/abstract ([pdf](https://www.aanda.org/articles/aa/pdf/2020/04/aa37222-19.pdf))

## Installation

`raccoon` is mainly implemented in python. It also uses some fortran subroutines that, in order to be called from python, are compiled with `f2py`.

Before installing it needs a python 3 distribution with `numpy`.

If you use conda you can create and activate a conda environment with:

```bash
conda create -n raccoon python=3.6 numpy
conda activate raccoon
```

### From source

The source code for `raccoon` can be downloaded from GitHub and installed by running

```bash
git clone https://github.com/mlafarga/raccoon.git
cd raccoon
python setup.py install
```

<!---
### Using pip

`raccoon` can be easily install using `pip` (after installing `numpy`)

```bash
pip install raccoon
```
-->


## Basic usage

After installation you will have available the different modules of the `raccoon` package and some scripts.
The scripts are the easiest way to use the package.

### Compute CCFs

CCFs can be computed with the script `raccoonccf`.

You need to specify the following 3 mandatory arguments
- `fil_or_list_spec`: The input spectra. This can be a file with the names of the reduced FITS spectra or directly the file names (names must include the absolute path to the files).
- `inst`: Instrument. `CARM_VIS`, `CARM_NIR`.
- `filmask`: Mask. You can choose your own mask file (include the absolute path) or use any of the available masks by specifing the mask id (see below).

There are also avaliable several optional arguments. Here is a description of the most important ones
- `--obj`: Name of the target. The oputput files use this name.
- `--rvabs`: The absolute RV of the target (in km/s) can be specified to locate the minimum of the CCF faster.
- `--ords_use`: List of the orders to consider. In general the following orders work well for these instruments:
    - `CARM_VIS`: `10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51`
    - `CARM_NIR`: `1 3 4 5 6 7 8 9 10 11 14 15 28 29 31 46 48 50 52`
- `--filtell`: Telluric mask file. To use the default masks: `--filtell default`
- `--rvshift`: Usually the spectra are not corrected for barycentric shifts or instrumental drifts. This argument specifies how to get these corrections.
- `--fcorrorders`: Same SED for all observations.

For all the available options see

```bash
raccoonccf -h
```

Here is a basic example of how to run the script with recommended options:
```bash
raccoonccf PATH/TO/OBS/car*vis_A.fits CARM_VIS J07274+052default --obj OBJ --filtell default --rvshift header --fcorrorders obshighsnr --ords_use 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 --plot_sv --verbose
```

Outputs:
- `spec_ccf.dat`: 
- `OBJ.ccfpar.dat`: 


### Build masks

Example for CARMENES VIS data:

```bash
raccoonmask PATH/TO/TPL.fits TPL_TYPE OBJ --inst CARM_VIS --tplrv 10. --cont poly --contfiltmed 1 --contfiltmax 400 --contpolyord 2 -line_fwhmmin 2.00 --line_fwhmmax 30.00 --line_contrastminmin 0.06 --line_depthw_percentdeepest 0.10 --line_depthw_depthmaxquantile 0.6 --verbose
```

For all the available options see

```bash
raccoonmask -h
```

Output:
- `TPL.mas`: 

## Available data

### Weighted binary masks

Mask available for CARMENES VIS and NIR:

| Mask ID              | SpT    | vsini [km/s] | Mask file                                                                                         |
| -------------------- | ------ | ------------ | ------------------------------------------------------------------------------------------------- |
| `J12123+544Sdefault` | M0.0 V |       <= 2.0 | `J12123+544Sserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas` |
| `J11033+359default`  | M1.5 V |       <= 2.0 | `J11033+359serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
| `J19169+051Ndefault` | M2.5 V |       <= 2.0 | `J19169+051Nserval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas` |
| `J07274+052default`  | M3.5 V |       <= 2.0 | `J07274+052serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
| `J13229+244default`  | M4.0 V |       <= 2.0 | `J13229+244serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
| `J20260+585default`  | M5.0 V |       <= 2.0 | `J20260+585serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
| `J10564+070default`  | M6.0 V |          2.9 | `J10564+070serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
| `J02530+168default`  | M7.0 V |       <= 2.0 | `J02530+168serval_tellbervmax_fwhm2.00-30.00_contrminmin0.06_depthwq0.60_contrastmeanfwhm-1.mas`  |
