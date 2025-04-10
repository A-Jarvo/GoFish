# GoFish
A simple and lightweight but user friendly Fisher forecast tool for DESI galaxy clustering. Adding ability to forecast phase shift due to free-streaming neutrinos and non-standard neutrino properties. Setting beta_phi_fixed = False will allow the phase shift amplitude for standard model neutrinos to vary, setting geff_fixed = False lets parameters controlling the functional form of the phase shift due to varying strength of neutrino self-interactions vary, giving a constraint on Geff where Geff controls the self-interaction strength for a simplified model.

You can specify your cosmology and the properties of your tracers in the configuration file.
You can take the 'config/test.ini' and 'input_files/DESI_BGS_nbar.txt'  that come with this package
and modify them to suit your needs.

The code will create a set of two output text files for each redshift: one for covariance and one for
the mean values (data).
The order of parameters in both the data and covaraince files is fs8, DA, H.
DA is in Megaparsecs and H is in Megaparsecs/km/s.

Now can add beta_phi for phase shift amplitude, in future planning to add parameters for non-standard neutrino properties to be measured.
