# GoFish
A simple and lightweight but user friendly Fisher forecast tool for galaxy clustering surveys with a variable number of tracers. 

Compared to the master branch, I have made several changes to this code, the main ones being adding ability to forecast constraints on the phase shift amplitude $$\beta_{\phi}$$ due to free-streaming neutrinos and the strength of neutrino self-interactions parameterized by $$\log_{10}{(G_{\mathrm{eff}})}$$ in a universal self-interaction model. 


# Installation 




# Running the code 


## Input file 

## Config file 

## Output file 



Setting beta_phi_fixed = False will allow the phase shift amplitude for standard model neutrinos to vary, setting geff_fixed = False lets parameters controlling the functional form of the phase shift due to varying strength of neutrino self-interactions vary, giving a constraint on Geff where Geff controls the self-interaction strength for a simplified model.
If you want to change the phase shift due to beta_phi, better off passing a value for beta_phi in the config file than changing Neff to accurately capture this effect. Leave Neff = 3.044.
Forecasts for log10Geff < -6 are likely to be inaccurate since the Fisher information is very small when the rate of change of the power spectrum w.r.t. Geff is effectively zero for these interaction strengths.
Ok if log10Geff is a fixed parameter when forecasting for other parameters.

You can specify your cosmology and the properties of your tracers in the configuration file.
You can take the 'config/test.ini' and 'input_files/DESI_BGS_nbar.txt'  that come with this package
and modify them to suit your needs.

The code will create a set of two output text files for each redshift: one for covariance and one for
the mean values (data).
The order of parameters in both the data and covaraince files is fs8, DA, H.
DA is in Megaparsecs and H is in Megaparsecs/km/s.
