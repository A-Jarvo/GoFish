import sys
import numpy as np
from configobj import ConfigObj
from TackleBox import Set_Bait, Fish, CovRenorm, shrink_sqr_matrix
from ioutils import CosmoResults, InputData, write_fisher
from rich.console import Console
from loguru import logger
from combined_forecasts_DESI import combined_forecasts_cross_correlations_DESI

if __name__ == "__main__":
    console = Console()
    # Read in the config file
    configfile = sys.argv[1]
    # print(sys.argv[1])
    pardict = ConfigObj(configfile)

    if "beta_phi_fixed" not in pardict:
        pardict["beta_phi_fixed"] = True
    if "geff_fixed" not in pardict:
        pardict["geff_fixed"] = True
    if "do_combined_DESI" not in pardict:
        pardict["do_combined_DESI"] = False

    if not pardict.as_bool("beta_phi_fixed") and not pardict.as_bool("BAO_only"):
        msg = "You have set beta_phi_fixed = False and BAO_only = False. This is not allowed."
        logger.error(msg)
        raise (ValueError)

    if not pardict.as_bool("geff_fixed") and not pardict.as_bool("BAO_only"):
        msg = (
            "You have set geff_fixed = False and BAO_only = False. This is not allowed."
        )
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
    if data.nbar is None:  # we got a file with nz instead of nbar, so do a conversion
        data.convert_nbar(cosmo.volume, float(pardict["skyarea"]))
        console.log("Number per redshift bin converted to number density per volume.")

    console.log(
        "Total number of objects:",
        np.nansum(
            np.array([data.nbar[i] * cosmo.volume for i in range(len(data.nbar))])
        ),
    )
    console.log("Total volume:", np.sum(cosmo.volume))

    # Scales the bias so that it goes as b/G(z)
    if pardict.as_bool("scale_bias"):
        data.scale_bias(cosmo.growth)
    console.log("#  Data nbar")
    console.log(data.nbar)
    console.log("#  Data bias")
    console.log(data.bias)

    console.log("Fitting beta_phi amplitude?")
    console.log((not pardict.as_bool("beta_phi_fixed")))

    console.log("Fitting geff?")
    console.log((not pardict.as_bool("geff_fixed")))

    # Precompute some things we might need for the Fisher matrix
    recon, derPalpha, derPalpha_BAO_only, derPbeta_amplitude, derPgeff = Set_Bait(
        cosmo,
        data,
        BAO_only=pardict.as_bool("BAO_only"),
        beta_phi_fixed=pardict.as_bool("beta_phi_fixed"),
        geff_fixed=pardict.as_bool("geff_fixed"),
    )
    console.log(
        "Computed reconstruction factors and derivatives of the power spectrum w.r.t. forecast parameters."
    )
    console.log("#  Data recon factor")
    console.log(recon)

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix

    erralpha = np.zeros(len(cosmo.z))
    FullCatch = np.zeros(
        (len(cosmo.z) * len(data.nbar) + 3, len(cosmo.z) * len(data.nbar) + 3)
    )
    # identity = np.eye(len(data.nbar) + 3)
    if not pardict.as_bool("beta_phi_fixed") and not pardict.as_bool("geff_fixed"):
        # identity = np.eye(len(data.nbar) + 5)
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)   beta_err(%)   log10Geff_err(%)"
        )
        erralpha = np.zeros(len(cosmo.z))
        FullCatch = np.zeros(
            (len(cosmo.z) * len(data.nbar) + 5, len(cosmo.z) * len(data.nbar) + 5)
        )
    elif not pardict.as_bool("beta_phi_fixed"):
        # identity = np.eye(len(data.nbar) + 4)
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)   beta_err(%)"
        )
        erralpha = np.zeros(len(cosmo.z))
        FullCatch = np.zeros(
            (len(cosmo.z) * len(data.nbar) + 4, len(cosmo.z) * len(data.nbar) + 4)
        )
    elif not pardict.as_bool("geff_fixed"):
        # identity = np.eye(len(data.nbar) + 4)
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)   geff_err(%)"
        )
        erralpha = np.zeros(len(cosmo.z))
        FullCatch = np.zeros(
            (len(cosmo.z) * len(data.nbar) + 4, len(cosmo.z) * len(data.nbar) + 4)
        )
    else:
        console.log(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)"
        )

    for iz in range(len(cosmo.z)):
        if np.any(data.nbar[:, iz] > 1.0e-30) and np.any(data.nz[:, iz] > 1.0e-30):
            Catch = Fish(
                cosmo,
                cosmo.kmin,
                cosmo.kmax,
                data,
                iz,
                recon[iz],
                derPalpha,
                derPbeta_amplitude,
                derPgeff,
                pardict.as_bool("BAO_only"),
                pardict.as_bool("GoFast"),
                pardict.as_bool("beta_phi_fixed"),
                pardict.as_bool("geff_fixed"),
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
                derPgeff,
                True,
                pardict.as_bool("GoFast"),
                pardict.as_bool("beta_phi_fixed"),
                pardict.as_bool("geff_fixed"),
            )

            if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
                Catch[-2:, -2:] += ExtraCatch[-2:, -2:]
            elif not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                Catch[-4:, -4:] += ExtraCatch[-4:, -4:]
            else:
                Catch[-3:, -3:] += ExtraCatch[-3:, -3:]

            # Add the Fisher matrix to the full fisher matrix
            FullCatch[
                iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                iz * len(data.nbar) : (iz + 1) * len(data.nbar),
            ] += Catch[: len(data.nbar), : len(data.nbar)]
            if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
                FullCatch[
                    iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                    -3:,
                ] += Catch[: len(data.nbar), -3:]
                FullCatch[-3:, iz * len(data.nbar) : (iz + 1) * len(data.nbar)] += (
                    Catch[-3:, : len(data.nbar)]
                )
                FullCatch[-3:, -3:] += Catch[-3:, -3:]
            if not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                FullCatch[
                    iz * len(data.nbar) : (iz + 1) * len(data.nbar),
                    -5:,
                ] += Catch[: len(data.nbar), -5:]
                FullCatch[-5:, iz * len(data.nbar) : (iz + 1) * len(data.nbar)] += (
                    Catch[-5:, : len(data.nbar)]
                )
                FullCatch[-5:, -5:] += Catch[-5:, -5:]
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
            # cov = dgesv(Catch, identity)[2]
            cov = np.linalg.inv(Catch)

            # Compute the error on isotropic alpha also
            J = np.array([2.0 / 3.0, 1.0 / 3.0])
            if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
                erralpha[iz] = 100.0 * np.sqrt(J @ cov[-2:, -2:] @ J.T)
            elif not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                erralpha[iz] = 100.0 * np.sqrt(J @ cov[-4:-2, -4:-2] @ J.T)
            else:
                erralpha[iz] = 100.0 * np.sqrt(J @ cov[-3:-1, -3:-1] @ J.T)

            # Renormalise the covariance from fsigma8, alpha_perp, alpha_par to fsigma8, Da, H
            means = np.array(
                [cosmo.f[iz] * cosmo.sigma8[iz], cosmo.da[iz], cosmo.h[iz]]
            )
            if not pardict.as_bool("beta_phi_fixed"):
                means = np.append(means, cosmo.beta_phi)
            if not pardict.as_bool("geff_fixed"):
                means = np.append(means, cosmo.log10Geff)
            cov_renorm = CovRenorm(
                cov,
                means,
                beta_phi_fixed=pardict.as_bool("beta_phi_fixed"),
                geff_fixed=pardict.as_bool("geff_fixed"),
            )

            # Print the parameter means and errors
            errs = None
            if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
                errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-3:]) / means
            elif not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-5:]) / abs(means)
            else:
                errs = 100.0 * np.sqrt(np.diag(cov_renorm)[-4:]) / abs(means)
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
                txt = txt + "       {0:.2f}".format(errs[3])
            if not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                txt = txt + "       {0:.2f}".format(errs[4])
            elif not pardict.as_bool("geff_fixed") and pardict.as_bool(
                "beta_phi_fixed"
            ):
                txt = txt + "       {0:.2f}".format(errs[3])
            console.log(txt)

            # Output the fisher matrix for the redshift bin
            write_fisher(
                pardict,
                cov_renorm,
                cosmo.z[iz],
                means,
                pardict.as_bool("beta_phi_fixed"),
                pardict.as_bool("geff_fixed"),
            )

        else:
            console.log(
                "Number density in this bin is zero, no data. Setting error on alpha to effectively infinite (disregard constraints on any parameters in this bin)."
            )
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
                txt = txt + "       {0:.2f}".format(errs[3])
            if not pardict.as_bool("geff_fixed") and not pardict.as_bool(
                "beta_phi_fixed"
            ):
                txt = txt + "       {0:.2f}".format(errs[4])
            elif not pardict.as_bool("geff_fixed") and pardict.as_bool(
                "beta_phi_fixed"
            ):
                txt = txt + "       {0:.2f}".format(errs[3])
            console.log(txt)

    # Combine the Fisher matrices
    cosmo = CosmoResults(
        pardict, np.atleast_1d(data.zmin[0]), np.atleast_1d(data.zmax[-1])
    )

    # Invert the Combined Fisher matrix to get the parameter
    # covariance matrix and compute means and errors

    flags = []
    FullCatchsmall = FullCatch.copy()
    for fi in np.arange(len(data.nz)):
        for fj in np.arange(len(data.nz[0])):
            if data.nz[fi][fj] <= 1.0e-25:
                flags.append(fi * len(data.nz) + fj)
    flags = np.array(flags)

    if len(flags) > 0:
        console.log("Removing rows for zero number density")
        FullCatchsmall = shrink_sqr_matrix(FullCatch, flags)

    if np.linalg.det(FullCatchsmall) == 0:
        console.log("Fisher (FullCatch) matrix is singular")
        console.log(
            "Checking if fisher information is zero for galaxy bias in any rows (must remove these)?"
        )
        flags = np.where(np.diag(FullCatchsmall) <= 1.0e-25)[0]
        if len(flags) > 0:
            if (
                np.sum(
                    np.where(np.array(flags) > (len(data.nbar) * len(data.nbar[0]) - 1))
                )
                > 0
            ):
                console.log(
                    "Fisher matrix is singular and information about some cosmo parameters of interest is zero."
                )
                console.log(
                    "This is a problem, and shouldn't happen, please check your input data."
                )
                console.log("Fisher matrix diagonals:")
                console.log(np.diag(FullCatchsmall))
                raise (ValueError)
            else:
                FullCatchsmall = shrink_sqr_matrix(FullCatchsmall)
                for i in range(len(flags)):
                    console.log(
                        "Fisher matrix is singular, removing row {0} and column {0}".format(
                            flags[i]
                        )
                    )
                    ntracer = flags[i] % len(
                        data.nbar
                    )  # the remainder here is the tracer number
                    zbin = flags[i] // len(
                        data.nbar
                    )  # the integer division here is the redshift bin number
                    console.log(
                        "Removed row for galaxy bias corresponding to tracer = {:d}, zbin = {:d}".format(
                            ntracer, zbin
                        )
                    )
                    console.log("Make sure this makes sense with your input data.")
        else:
            console.log("Fisher matrix is singular and no zero rows found.")
            console.log("This is a problem - please check your input data.")
            console.log("Fisher matrix diagonals:")
            console.log(np.diag(FullCatchsmall))
            raise (ValueError)

    covFull = np.linalg.inv(FullCatchsmall)

    J = np.array([2.0 / 3.0, 1.0 / 3.0])
    erralpha = None
    if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
        erralpha = 100.0 * np.sqrt(J @ covFull[-2:, -2:] @ J.T)
    elif not pardict.as_bool("beta_phi_fixed") and not pardict.as_bool("geff_fixed"):
        erralpha = 100.0 * np.sqrt(J @ covFull[-4:-2, -4:-2] @ J.T)
    else:
        erralpha = 100.0 * np.sqrt(J @ covFull[-3:-1, -3:-1] @ J.T)

    means = np.array([cosmo.f[0] * cosmo.sigma8[0], cosmo.da[0], cosmo.h[0]])
    if not pardict.as_bool("beta_phi_fixed"):
        means = np.append(means, cosmo.beta_phi)
    if not pardict.as_bool("geff_fixed"):
        means = np.append(means, cosmo.log10Geff)

    cov_renormFull = CovRenorm(
        covFull,
        means,
        beta_phi_fixed=pardict.as_bool("beta_phi_fixed"),
        geff_fixed=pardict.as_bool("geff_fixed"),
    )

    write_fisher(
        pardict,
        cov_renormFull,
        1e30,
        means,
        pardict.as_bool("beta_phi_fixed"),
        pardict.as_bool("geff_fixed"),
    )

    errs = None
    if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
        errs = 100.0 * np.sqrt(np.diag(cov_renormFull)[-3:]) / means
    elif not pardict.as_bool("geff_fixed") and not pardict.as_bool("beta_phi_fixed"):
        errs = 100.0 * np.sqrt(np.diag(cov_renormFull)[-5:]) / abs(means)
    else:
        errs = 100.0 * np.sqrt(np.diag(cov_renormFull)[-4:]) / abs(means)
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
        txt = txt + "       {0:.2f}".format(errs[3])
    if not pardict.as_bool("geff_fixed") and not pardict.as_bool("beta_phi_fixed"):
        txt = txt + "       {0:.2f}".format(errs[4])
    elif not pardict.as_bool("geff_fixed") and pardict.as_bool("beta_phi_fixed"):
        txt = txt + "       {0:.2f}".format(errs[3])
    console.log(txt)

    cov_main = None
    if pardict["do_combined_DESI"] == "True":
        # Compute the cross-correlations between the DESI tracers
        fisher_main = combined_forecasts_cross_correlations_DESI(
            pardict,
            data,
            beta_phi_fixed=pardict.as_bool("beta_phi_fixed"),
            geff_fixed=pardict.as_bool("geff_fixed"),
        )

        # get the forecasts for fsigma8, beta_phi, geff

        cov_main = np.linalg.inv(fisher_main)

        means_main = np.array(
            [cosmo.f[0] * cosmo.sigma8[0], cosmo.beta_phi, cosmo.log10Geff]
        )

        errs = None
        if pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
            errs = 100.0 * np.sqrt(np.diag(cov_main)[-1:]) / means[-1:]
        elif not pardict.as_bool("geff_fixed") and not pardict.as_bool(
            "beta_phi_fixed"
        ):
            errs = 100.0 * np.sqrt(np.diag(cov_main)[-3:]) / abs(means[-3:])
        else:
            errs = 100.0 * np.sqrt(np.diag(cov_main)[-2:]) / abs(means[-2:])
        console.log("#  Combined errors for DESI forecasts")
        console.log("#=================")
        txt = " {0:.2f}    {1:.4f}     {2:.3f}       {3:.2f}".format(
            cosmo.z[0],
            cosmo.volume[0] / 1e9,
            means[0],
            errs[0],
        )
        if not pardict.as_bool("beta_phi_fixed"):
            console.log("#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)   beta_phi_err(%)")
            txt = txt + "       {0:.2f}".format(errs[1])
        if not pardict.as_bool("geff_fixed") and not pardict.as_bool("beta_phi_fixed"):
            console.log(
                "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)   beta_phi_err(%)   log10Geff_err(%)"
            )
            txt = txt + "       {0:.2f}".format(errs[2])
        elif not pardict.as_bool("geff_fixed") and pardict.as_bool("beta_phi_fixed"):
            console.log("#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)   log10Geff_err(%)")
            txt = txt + "       {0:.2f}".format(errs[1])
        else:
            console.log("#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)")
        console.log(txt)

        # make some pretty contour plots
        if not pardict.as_bool("beta_phi_fixed") and pardict.as_bool("geff_fixed"):
            # plot the contour for beta_phi and alpha_iso
            from chainconsumer import ChainConsumer, Chain
            import matplotlib.pyplot as plt

            covwanted = cov_main[-10:, -10:]
            means = np.array(
                [
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.f[0] * cosmo.sigma8[0],
                    cosmo.beta_phi,
                ]
            )
            c = ChainConsumer()
            c.add_chain(
                Chain.from_covariance(
                    means,
                    covwanted,
                    columns=[
                        r"$D(z_1)$",
                        r"$H(z_1)$",
                        r"$D(z_2)$",
                        r"$H(z_2)$",
                        r"$D(z_3)$",
                        r"$H(z_3)$",
                        r"$D(z_4)$",
                        r"$H(z_4)$",
                        r"f(z)\sigma_8(z)",
                        r"$\beta_{\phi}$",
                    ],
                    name="cov",
                )
            )
            c.plotter.plot()
            plt.show()

        elif not pardict.as_bool("geff_fixed") and pardict.as_bool("beta_phi_fixed"):
            # plot the contour for beta_phi and alpha_iso
            from chainconsumer import ChainConsumer, Chain
            import matplotlib.pyplot as plt

            covwanted = cov_main[-10:, -10:]
            means = np.array(
                [
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.f[0] * cosmo.sigma8[0],
                    cosmo.log10Geff,
                ]
            )
            c = ChainConsumer()
            c.add_chain(
                Chain.from_covariance(
                    means,
                    covwanted,
                    columns=[
                        r"$D(z_1)$",
                        r"$H(z_1)$",
                        r"$D(z_2)$",
                        r"$H(z_2)$",
                        r"$D(z_3)$",
                        r"$H(z_3)$",
                        r"$D(z_4)$",
                        r"$H(z_4)$",
                        r"f(z)\sigma_8(z)",
                        r"$\log_{10}G_{\mathrm{eff}}$",
                    ],
                    name="cov",
                )
            )
            c.plotter.plot()
            plt.show()

        elif not pardict.as_bool("geff_fixed") and not pardict.as_bool(
            "beta_phi_fixed"
        ):
            # plot the contour for beta_phi and alpha_iso
            from chainconsumer import ChainConsumer, Chain
            import matplotlib.pyplot as plt

            covwanted = cov_main[-11:, -11:]
            means = np.array(
                [
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.h[0],
                    cosmo.da[0],
                    cosmo.f[0] * cosmo.sigma8[0],
                    cosmo.log10Geff,
                ]
            )
            c = ChainConsumer()
            c.add_chain(
                Chain.from_covariance(
                    means,
                    covwanted,
                    columns=[
                        r"$D(z_1)$",
                        r"$H(z_1)$",
                        r"$D(z_2)$",
                        r"$H(z_2)$",
                        r"$D(z_3)$",
                        r"$H(z_3)$",
                        r"$D(z_4)$",
                        r"$H(z_4)$",
                        r"f(z)\sigma_8(z)",
                        r"$\beta_{\phi}$",
                        r"$\log_{10}G_{\mathrm{eff}}$",
                    ],
                    name="cov",
                )
            )
            c.plotter.plot()
            plt.show()
