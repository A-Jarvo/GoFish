import sys
import numpy as np
from configobj import ConfigObj
from TackleBox import Set_Bait, Fish, CovRenorm
from ioutils import CosmoResults, InputData, write_fisher
from scipy.linalg.lapack import dgesv
from rich.console import Console
from loguru import logger

if __name__ == "__main__":
    console = Console()
    # Read in the config file
    configfile = sys.argv[1]
    # print(sys.argv[1])
    pardict = ConfigObj(configfile)

    if "beta_phi_fixed" not in pardict:
        pardict["beta_phi_fixed"] = True

    if not pardict.as_bool("beta_phi_fixed") and not pardict.as_bool("BAO_only"):
        msg = "You have set beta_phi_fixed = False and BAO_only = False. This is not allowed."
        logger.error(msg)
        raise (ValueError)

    # Read in the file containing the redshift bins, nz and bias values
    data = InputData(pardict)
    console.log("Read in the data file for redshifts, number density and bias.")

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(pardict, data.zmin, data.zmax)
    if np.any(data.volume > 0):
        cosmo.volume = data.volume * 1.0e9

    console.log("computed CAMB linear matter power spectra for redshifts.")

    # Convert the nz to nbar in (h/Mpc)^3
    data.convert_nbar(cosmo.volume, float(pardict["skyarea"]))

    console.log("Number per redshift bin converted to number density per volume.")

    print(
        "Total number of objects:",
        np.sum(np.array([data.nbar[i] * cosmo.volume for i in range(len(data.nbar))])),
    )
    print("Total volume:", np.sum(cosmo.volume))

    # Scales the bias so that it goes as b/G(z)
    if pardict.as_bool("scale_bias"):
        data.scale_bias(cosmo.growth)
    console.log("#  Data nbar")
    console.log(data.nbar)
    console.log("#  Data bias")
    console.log(data.bias)

    console.log("Fitting beta_phi amplitude?")
    console.log(not pardict["beta_phi_fixed"])

    # Precompute some things we might need for the Fisher matrix
    recon, derPalpha, derPalpha_BAO_only, derPbeta_amplitude = Set_Bait(
        cosmo,
        data,
        BAO_only=pardict.as_bool("BAO_only"),
        beta_phi_fixed=pardict.as_bool("beta_phi_fixed"),
    )
    console.log(
        "Computed reconstruction factors and derivatives of the power spectrum w.r.t. forecast parameters."
    )
    console.log("#  Data recon factor")
    console.log(recon)

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix

    if not pardict.as_bool("beta_phi_fixed"):
        identity = np.eye(len(data.nbar) + 4)
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)   beta_err(%)"
        )
        erralpha = np.zeros(len(cosmo.z))
        FullCatch = np.zeros(
            (len(cosmo.z) * len(data.nbar) + 4, len(cosmo.z) * len(data.nbar) + 4)
        )
    else:
        identity = np.eye(len(data.nbar) + 3)
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)"
        )
        erralpha = np.zeros(len(cosmo.z))
        FullCatch = np.zeros(
            (len(cosmo.z) * len(data.nbar) + 3, len(cosmo.z) * len(data.nbar) + 3)
        )

    for iz in range(len(cosmo.z)):
        if np.any(data.nz[:, iz] > 1.0e-30):
            Catch = Fish(
                cosmo,
                cosmo.kmin,
                cosmo.kmax,
                data,
                iz,
                recon[iz],
                derPalpha,
                derPbeta_amplitude,
                pardict.as_bool("BAO_only"),
                pardict.as_bool("GoFast"),
                pardict.as_bool("beta_phi_fixed"),
            )

            # Add on BAO only information from kmax to k = 0.5 Mpc/h but only for alpha_perp and alpha_par
            ExtraCatch = Fish(
                cosmo,
                cosmo.kmax,
                0.5,
                data,
                iz,
                recon[iz],
                derPalpha_BAO_only,
                derPbeta_amplitude,
                True,
                pardict.as_bool("GoFast"),
                pardict.as_bool("beta_phi_fixed"),
            )

            if pardict.as_bool("beta_phi_fixed"):
                Catch[-2:, -2:] += ExtraCatch[-2:, -2:]
            else:
                Catch[-3:, -3:] += ExtraCatch[-3:, -3:]

            # Add the Fisher matrix to the full fisher matrix
            FullCatch[
                iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                iz * len(data.nbar) : (iz + 1) * len(data.nbar),
            ] += Catch[: len(data.nbar), : len(data.nbar)]
            if pardict.as_bool("beta_phi_fixed"):
                FullCatch[
                    iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                    -3:,
                ] += Catch[: len(data.nbar), -3:]
                FullCatch[-3:, iz * len(data.nbar) : (iz + 1) * len(data.nbar)] += (
                    Catch[-3:, : len(data.nbar)]
                )
                FullCatch[-3:, -3:] += Catch[-3:, -3:]
            else:
                FullCatch[
                    iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                    -4:,
                ] += Catch[: len(data.nbar), -4:]
                FullCatch[-4:, iz * len(data.nbar) : (iz + 1) * len(data.nbar)] += (
                    Catch[-4:, : len(data.nbar)]
                )
                FullCatch[-4:, -4:] += Catch[-4:, -4:]

            # Invert the Fisher matrix to get the parameter covariance matrix
            cov = dgesv(Catch, identity)[2]

            # Compute the error on isotropic alpha also
            J = np.array([2.0 / 3.0, 1.0 / 3.0])
            if pardict.as_bool("beta_phi_fixed"):
                erralpha[iz] = 100.0 * np.sqrt(J @ cov[-2:, -2:] @ J.T)
            else:
                erralpha[iz] = 100.0 * np.sqrt(J @ cov[-3:-1, -3:-1] @ J.T)

            # Renormalise the covariance from fsigma8, alpha_perp, alpha_par to fsigma8, Da, H
            means = np.array(
                [cosmo.f[iz] * cosmo.sigma8[iz], cosmo.da[iz], cosmo.h[iz]]
            )
            if not pardict.as_bool("beta_phi_fixed"):
                means = np.append(means, cosmo.beta_phi)
            cov_renorm = CovRenorm(
                cov, means, beta_phi_fixed=pardict.as_bool("beta_phi_fixed")
            )

            # print(Catch)

            # Print the parameter means and errors
            errs = None
            if pardict.as_bool("beta_phi_fixed"):
                errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-3:]) / means
            else:
                errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-4:]) / means
            txt = " {0:.2f}    {1:.4f}     {2:.3f}       {3:.2f}         {4:.1f}       {5:.2f}        {6:.1f}       {7:.2f}       {8:.3f}".format(
                cosmo.z[iz],
                cosmo.volume[iz] / 1e9,
                means[0],
                errs[0],
                means[1],
                errs[1],
                means[2],
                errs[2],
                erralpha[iz],
            )
            if not pardict.as_bool("beta_phi_fixed"):
                # txt = txt + "       {0:.2f}".format(means[3])
                txt = txt + "       {0:.2f}".format(errs[3])
            console.log(txt)

            # Output the fisher matrix for the redshift bin
            write_fisher(
                pardict,
                cov_renorm,
                cosmo.z[iz],
                means,
                pardict.as_bool("beta_phi_fixed"),
            )

        else:
            erralpha[iz] = 1.0e30
            # " {0:.2f}     {1:.4f}    {2:.3f}         -          {4:.1f}         -         {6:.1f}         -          -".format(
            txt = " {0:.2f}    {1:.4f}     {2:.3f}       {3:.2f}         {4:.1f}       {5:.2f}        {6:.1f}       {7:.2f}       {8:.3f}".format(
                cosmo.z[iz],
                cosmo.volume[iz] / 1e9,
                means[0],
                errs[0],
                means[1],
                errs[1],
                means[2],
                errs[2],
                erralpha[iz],
            )
            if not pardict.as_bool("beta_phi_fixed"):
                # txt = txt + "       {0:.2f}".format(means[3])
                txt = txt + "       {0:.2f}".format(errs[3])
            console.log(txt)

    # Run the cosmological parameters at the centre of the combined redshift bin
    identity = np.eye(len(cosmo.z) * len(data.nbar) + 3)
    if not pardict.as_bool("beta_phi_fixed"):
        identity = np.eye(len(cosmo.z) * len(data.nbar) + 4)
    # Combine the Fisher matrices
    cosmo = CosmoResults(
        pardict, np.atleast_1d(data.zmin[0]), np.atleast_1d(data.zmax[-1])
    )

    # Invert the Combined Fisher matrix to get the parameter
    # covariance matrix and compute means and errors
    cov = dgesv(FullCatch, identity)[2]
    J = np.array([2.0 / 3.0, 1.0 / 3.0])
    erralpha = 100.0 * np.sqrt(J @ cov[-2:, -2:] @ J.T)
    if not pardict.as_bool("beta_phi_fixed"):
        erralpha = 100.0 * np.sqrt(J @ cov[-3:-1, -3:-1] @ J.T)
    means = np.array([cosmo.f[0] * cosmo.sigma8[0], cosmo.da[0], cosmo.h[0]])
    if not pardict.as_bool("beta_phi_fixed"):
        means = np.append(means, cosmo.beta_phi)
    cov_renorm = CovRenorm(cov, means, beta_phi_fixed=pardict.as_bool("beta_phi_fixed"))
    errs = None
    if pardict.as_bool("beta_phi_fixed"):
        errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-3:]) / means
    else:
        errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-4:]) / means
    console.log("#  Combined errors")
    console.log("#=================")
    txt = " {0:.2f}    {1:.4f}     {2:.3f}       {3:.2f}         {4:.1f}       {5:.2f}        {6:.1f}       {7:.2f}       {8:.3f}".format(
        cosmo.z[0],
        cosmo.volume[0] / 1e9,
        means[0],
        errs[0],
        means[1],
        errs[1],
        means[2],
        errs[2],
        erralpha,
    )
    if not pardict.as_bool("beta_phi_fixed"):
        # txt = txt + "       {0:.2f}".format(means[3])
        txt = txt + "       {0:.2f}".format(errs[3])
    console.log(txt)
