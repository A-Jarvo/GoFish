# GoFish

A simple and lightweight but user friendly Fisher forecast tool for galaxy clustering surveys with a variable number of tracers. 

The code is able to forecast the constraining power on $$f\sigma_8$$ (the growth rate $$\times$$ the variance of matter fluctuations on scales of 8/h Mpc), and a measurement of angular diameter distance $$D_A(z)$$ and $$H(z)$$ by interpreting the BAO/AP effect distortion parameters as the ratios $$q_{\parallel} = \frac{H^{\mathrm{fiducial}}(z)}{H(z)}$$ and $$q_{\perp} = \frac{D_A(z)}{D^{\mathrm{fiducial}}_A(z)}$$.

Compared to the master branch, I have made several changes to this code, the main ones being adding ability to forecast constraints on the phase shift amplitude $$\beta_{\phi}$$ due to free-streaming neutrinos and the strength of neutrino self-interactions parameterized by $$\log_{10}{(G_{\mathrm{eff}})}$$ in a universal self-interaction model. There is an ability to switch these options off if the user is not interested in these parameters. 

All power spectra used for forecasts in this code relies on CAMB. 

The code will treat the galaxy bias $$\times \sigma_8$$ as a separate variable in each redshift bin, and likewise $$f\sigma_8$$, $$D_A(z)$$ and $$H(z)$$. The forecasts for the latter three parameters are output to the user in each redshift bin provided in the input data file (this is discussed below in more detail). However, the full covariance matrix in each redshift bin and saved for the user to access later. An output for the covariance obtained from summing the fisher information aggregated over all redshift bins for $$f\sigma_8$$, $$D_A(z)$$ and $$H(z)$$ is also output to the user.  

# Installation 

This code can be run by simply running the Makefile, like 'make install'. This file will install a UV virtual environment, and ensure the user has the required packages to run the script, which are listed in .pyproject.toml. Within the UV environment, it is simple to run commands like 'uv sync' to update packages and 'uv run ..' to run programs within the environment. 


# Running the code 

Running the script involves preparation of a config file with the suffix .ini and an input file. The config file specifies settings for the forecasts and the input file contains information about the galaxy survey forecasts are being produced for. 

The script can be run in this manner, where the config file name is used as the first command line argument: 

```uv run GoFish.py ../config/config.ini ```

## Input file 

The input file essential information for determining the signal/noise of parameter constraints. It should contain the number density of objects in each redshift bin (see the example input files in input/). For columns with the name 'dn_dz', the code will interpret the data as the number of objects per dz per square degree. If the column is named 'nbar', it will be interpreted as the number density of objects in units of number * h per mpc (h/megaparsec) cubed $$\times 10^3$$. It is not necessary to provide the survey volume for each redshift bin in this file. It is necessary to provide the galaxy bias in each redshift bin, which can optionally be scaled with the growth rate in each redshift bin (specified in the config file) if it is not done so in the input file.  If multiple columns for dn_dz and galaxy bias are provided in the input file, the code will treat each additional column as data for an additional tracer. In that way, the user can specify to produce forecasts for as many tracers as desired. 

## Config file 

The config file allows the user to specify the fiducial cosmology for the forecasts, and some other settings related to the galaxy survey data. Please see some of the example config files to see how to specify different cosmological parameters. One can also set kmax, kmin for the survey forecasts, and the sky area for the forecasts. It is also possible to specify whether $$\beta_{\phi}$$ for the phase shift should be fixed, 

```beta_phi_fixed = True ```

or likewise $$G_{\mathrm{eff}}$$,

``` geff_fixed = True ```  

If you want to change the phase shift amplitude due to neutrinos, in the configuration file, the user should specify the value of ```beta_phi``` in the config file than changing Neff. Leave Neff = 3.044.
Forecasts for log10Geff < -6 are likely to be inaccurate since the Fisher information is very small when the rate of change of the power spectrum w.r.t. Geff is effectively zero for these interaction strengths and this may lead to a singular Fisher matrix if ```geff_fixed = False ```. 
It is ok for it to take a small value if it is a fixed parameter, ``` log10Geff = -12.0 ```. 

The setting ```BAO_only``` changes the forecast information from constraints from the oscillatory signal only, rather than using the full shape power spectrum. The setting ```GoFast``` should be left as True, changing it will lead the code to stop because it is a setting related to some now deprecated code that has not been updated. 

## Output file 

The code will create a set of two output text files for each redshift: one for covariance and one for
the mean values (data). The prefix for the names of the input files in specified in the config file. 


# Optional extra for DESI specific forecasts 

I wrote an additional script that will be run in the case one adds this to the ini file:

```do_combined_DESI = True```

This script will output uncertainties on each cosmological parameter from a combined fisher forecast where every parameter is treated separately in each bin (rather than aggregating $$f\sigma_8$$, $$D_A(z)$$ and $$H(z)$$ across each bin). 

This script explicity does this by aggregating information from redshifts of 0-0.4 (BGS), 0.4-0.6 (LRG1), 0.6-1.1 (LRG2), 1.1-1.7 (LRG3/QSOs) and 1.7 - 2.1 (QSOs). The resulting fisher matrix will allow for a covariance matrix for $$f\sigma_8$$, $$D_A(z)$$ and $$H(z)$$ in these 5 bins (and also the optional pahse shift parameters if desired). 


