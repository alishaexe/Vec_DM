# VecDM Readme

## Numerical calculation files
The files
1. ```JaxIntegration.py```
2. ```transfer_plot.py```

Both of these have output files in the ```/datafiles``` folder which can be used for any plots, as they both take a while to run ~ 7 minutes for JaxIntegration.

## No computation files
The files
1. ```CompareTransfer.py```
2. ```OmegGW_NoComps.py```

Both of these can be run without jax/other packages except numpy, matplotlib, and scipy. They just use the files from ```/datafiles``` to make the plots.
The function for the analytical fit for our model is in ```OmegGW_NoComps.py```.

## Datafiles
The files
1. ```jaxOmeg.npy```
2. ```PS.npy```
3. ```characteristic_strain_noise_spectra_NG15yr_psrs.txt```
4. ```residual_noise_power_spectra_NG15yr_psrs.txt```
5. ```sensitivity_curves_NG15yr_fullPTA.txt```

### ```PS.npy```
Gives the transfer function x<sup>2</sup><sub>*</sub> |T|<sup>2</sup>, for 
- xsFIN = 500
- xsIN = 0.001
- xsnum = 500
- xs = xsIN * (xsFIN/xsIN) ** ((np.arange(xsnum)) / xsnum)

this file doesn't contain any xs values (might change in future)

### ```jaxOmeg.npy```
Gives the Numerical data for \Omega<sub>GW</sub>, contains xt, and Omega_GW. produced by the ```JaxIntegration.py``` script. 

### Nanograv data in data files

This is copy-pasted straight from the nanograv readme.md file
This repository contains noise spectra for individual pulsars and stochastic gravitational wave background sensitivity curves for the NANOGrav 
15-year data set analysis, highlighted in the paper "The NANOGrav 15-Year Data Set: Detector Characterization and Noise Budget". As in the 
paper these spectra include the noise recovered from a common uncorrelated process analysis across the entire PTA. In other words the spectra 
include the white noise, the power from the common process and any significant additional red noise in an individual pulsar. The three files 
contain the following files of spectra:

1. 'characteristic_strain_noise_spectra_NG15yr_psrs.txt' contains a comma separated table of characteristic noise spectra for the 67 pulsars used 
in the 15-year gravitational wave analysis. The header lists the pulsar names in the various columns. The first column is the list of frequencies at 
which the characteristic strain is evaluated.

2. 'residual_noise_power_spectra_NG15yr_psrs.txt' contains a comma separated table of residual noise power spectra for the 67 pulsars used in the 15-year gravitational wave analysis. The header lists the pulsar names in the various columns. The first column is the list of 
frequencies at which the power spectral density is evaluated.

3. 'sensitivity_curves_NG15yr_fullPTA.txt' contains stochastic gravitational wave background sensitivity curves for the full array of pulsars in 
the PTA. The header lists the contents of the various columns. The first column is the frequencies where the curves were evaluated. The second 
column is the characteristic strain, the third column is the strain power spectral density and the last column is the ratio of cosmological 
energy density, calculated using H0=67.4 km/s/Mpc to match the paper  
