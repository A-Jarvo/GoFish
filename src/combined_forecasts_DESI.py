import numpy as np
from configobj import ConfigObj
from ioutils import InputData


def combined_forecasts_cross_correlations_DESI(
    pardict: ConfigObj,
    data: InputData,
    beta_phi_fixed: bool = True,
    geff_fixed: bool = True,
):
    # shared_bins = {
    #     "BGS": [0.0, 0.4, 4],  # zmin, zmax, number bins being joined
    #     "LRG1": [0.4, 0.5, 1],
    #     "LRG2": [0.6, 0.8, 2],
    #     "LRG3_ELG1": [0.8, 1.1, 3],
    #     "QSOs": [1.1, 1.9, 8],
    # }

    shared_bins = {  # validation paper
        "BGS": [0.0, 0.4, 4],  # zmin, zmax, number bins being joined
        "LRG1": [0.4, 0.6, 2],
        "LRG2": [0.6, 1.1, 5],
        "LRG3_ELG1": [1.1, 1.6, 5],
        "QSOs": [1.6, 2.1, 5],
    }

    num_params = (
        len(shared_bins) * 4
    )  # each z/tracer bin for bsigma8, alpha para, alpha perp, fsigma8
    if not beta_phi_fixed:
        num_params += 1
    if not geff_fixed:
        num_params += 1

    num = 0
    if not beta_phi_fixed:
        num += 1
    if not geff_fixed:
        num += 1

    fisher_main = np.zeros((num_params, num_params))

    # loop through the shared bins and read in precomputed fisher information, add together as appropriate
    for b, bin_name in enumerate(shared_bins):
        z_list = np.linspace(
            shared_bins[bin_name][0],
            shared_bins[bin_name][1],
            shared_bins[bin_name][2] + 1,
        )

        zmid = (z_list[1:] + z_list[:-1]) / 2.0

        # each matrix has [bsigma8_z1, bsigma8_z2, bsigma8_z3, bsigma8_z4, fsigma8, da, H, beta_phi, geff] (last two optional) - we want to rearrange to collapse bsigma8 in sets of bins and have
        # fsigma8, da, H, beta_phi, geff in the same order as the shared bins (but beta_phi and geff will be collapsed across all bins)
        for i, z in enumerate(zmid):
            # read in the precomputed fisher information for this bin

            cov_this_bin = pardict["outputfile"] + "_cov_" + format(z, ".2f") + ".txt"

            fisher = np.linalg.inv(np.loadtxt(cov_this_bin))

            fisher_shape_tracers = fisher.shape[0]
            if not beta_phi_fixed:
                fisher_shape_tracers -= 1
            if not geff_fixed:
                fisher_shape_tracers -= 1
            fisher_shape_bsigma8 = (
                fisher_shape_tracers - 3
            )  # subtract 3 for fsigma8, da, H
            index_fsigma8 = fisher_shape_bsigma8
            index_da = fisher_shape_bsigma8 + 1
            index_H = fisher_shape_bsigma8 + 2

            # sorting out diagonals first
            fisher_main[b * 4 : (b * 4 + 1), b * 4 : (b * 4 + 1)] += np.sum(
                np.array([fisher[i, i] for i in np.arange(fisher_shape_bsigma8)])
            )  # b * sigma8 - collapsed from all tracers in these bins
            fisher_main[b * 4 + 1 : (b * 4 + 1) + 1, b * 4 + 1 : (b * 4 + 1) + 1] += (
                fisher[index_fsigma8, index_fsigma8]
            )  # fsigma8
            fisher_main[b * 4 + 2 : (b * 4 + 1) + 2, b * 4 + 2 : (b * 4 + 1) + 2] += (
                fisher[index_da, index_da]
            )  # da
            fisher_main[b * 4 + 3 : (b * 4 + 1) + 3, b * 4 + 3 : (b * 4 + 1) + 3] += (
                fisher[index_H, index_H]
            )  # H

            if (
                not beta_phi_fixed or not geff_fixed
            ):  # also takes care of cross-terms of beta_phi and geff with each other
                fisher_main[-num:, -num:] += fisher[-num:, -num:]

            # cross terms with bsigma8

            fisher_main[b * 4 : (b * 4 + 1), b * 4 + 1 : (b * 4 + 1) + 1] += np.sum(
                np.array(
                    [fisher[i, index_fsigma8] for i in np.arange(fisher_shape_bsigma8)]
                )
            )  # cross bsigma8 and fsigma8
            fisher_main[b * 4 + 1 : (b * 4 + 1) + 1, b * 4 : (b * 4 + 1)] += np.sum(
                np.array(
                    [fisher[index_fsigma8, i] for i in np.arange(fisher_shape_bsigma8)]
                )
            )  # cross bsigma8 and fsigma8

            fisher_main[b * 4 : (b * 4 + 1), b * 4 + 2 : (b * 4 + 1) + 2] += np.sum(
                np.array([fisher[i, index_da] for i in np.arange(fisher_shape_bsigma8)])
            )  # cross bsigma8 and da
            fisher_main[b * 4 + 2 : (b * 4 + 1) + 2, b * 4 : (b * 4 + 1)] += np.sum(
                np.array([fisher[index_da, i] for i in np.arange(fisher_shape_bsigma8)])
            )  # cross bsigma8 and da

            fisher_main[b * 4 : (b * 4 + 1), b * 4 + 3 : (b * 4 + 1) + 3] += np.sum(
                np.array([fisher[i, index_H] for i in np.arange(fisher_shape_bsigma8)])
            )  # cross bsigma8 and H
            fisher_main[b * 4 + 3 : (b * 4 + 1) + 3, b * 4 : (b * 4 + 1)] += np.sum(
                np.array([fisher[index_H, i] for i in np.arange(fisher_shape_bsigma8)])
            )  # cross bsigma8 and H

            if not beta_phi_fixed or not geff_fixed:
                fisher_main[b * 4 : (b * 4 + 1), -num:] += np.sum(
                    np.array(
                        [fisher[i, -num:] for i in np.arange(fisher_shape_bsigma8)]
                    )
                )
                fisher_main[-num:, b * 4 : (b * 4 + 1)] += np.sum(
                    np.array(
                        [fisher[-num:, i] for i in np.arange(fisher_shape_bsigma8)]
                    )
                )  # cross bsigma8 and beta_phi/geff

            # cross terms with fsigma8

            fisher_main[b * 4 + 1 : (b * 4 + 1) + 1, b * 4 + 2 : (b * 4 + 1) + 2] += (
                fisher[index_fsigma8, index_da]
            )  # cross fsigma8 and da
            fisher_main[b * 4 + 2 : (b * 4 + 1) + 2, b * 4 + 1 : (b * 4 + 1) + 1] += (
                fisher[index_da, index_fsigma8]
            )  # cross fsigma8 and da

            fisher_main[b * 4 + 1 : (b * 4 + 1) + 1, b * 4 + 3 : (b * 4 + 1) + 3] += (
                fisher[index_fsigma8, index_H]
            )  # cross fsigma8 and H
            fisher_main[b * 4 + 3 : (b * 4 + 1) + 3, b * 4 + 1 : (b * 4 + 1) + 1] += (
                fisher[index_H, index_fsigma8]
            )  # cross fsigma8 and H

            if not beta_phi_fixed or not geff_fixed:
                fisher_main[b * 4 + 1 : (b * 4 + 1) + 1, -num:] += fisher[
                    index_fsigma8, -num:
                ]
                fisher_main[-num:, b * 4 + 1 : (b * 4 + 1) + 1] += fisher[
                    -num:, index_fsigma8
                ]  # cross fsigma8 and beta_phi/geff

            # cross terms with da
            fisher_main[b * 4 + 2 : (b * 4 + 1) + 2, b * 4 + 3 : (b * 4 + 1) + 3] += (
                fisher[index_da, index_H]
            )  # cross da and H
            fisher_main[b * 4 + 3 : (b * 4 + 1) + 3, b * 4 + 2 : (b * 4 + 1) + 2] += (
                fisher[index_H, index_da]
            )

            if not beta_phi_fixed or not geff_fixed:
                fisher_main[b * 4 + 2 : (b * 4 + 1) + 2, -num:] += fisher[
                    index_da, -num:
                ]  # cross da and beta_phi/geff
                fisher_main[-num:, b * 4 + 2 : (b * 4 + 1) + 2] += fisher[
                    -num:, index_da
                ]

            # cross terms with H
            if not beta_phi_fixed or not geff_fixed:
                fisher_main[b * 4 + 3 : (b * 4 + 1) + 3, -num:] += fisher[
                    index_H, -num:
                ]  # cross H and beta_phi/geff
                fisher_main[-num:, b * 4 + 3 : (b * 4 + 1) + 3] += fisher[
                    -num:, index_H
                ]

    return fisher_main
