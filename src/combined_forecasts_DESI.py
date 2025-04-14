import numpy as np
from configobj import ConfigObj
from ioutils import InputData


def combined_forecasts_cross_correlations_DESI(
    pardict: ConfigObj,
    data: InputData,
    beta_phi_fixed: bool = True,
    geff_fixed: bool = True,
):
    shared_bins = {
        "BGS": [0.0, 0.4, 4],  # zmin, zmax, number bins being joined
        "LRG1": [0.4, 0.6, 2],
        "LRG2": [0.6, 0.8, 2],
        "LRG3_ELG1": [0.8, 1.1, 3],
        "QSOs": [1.1, 1.9, 8],
    }

    num_params = (
        len(shared_bins) * 3 + 1
    )  # each z/tracer bin for bsigma8, alpha para, alpha perp then + 1 for fsigma8
    if not beta_phi_fixed:
        num_params += 1
    if not geff_fixed:
        num_params += 1

    num = 1
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

        for i, z in enumerate(zmid):
            # read in the precomputed fisher information for this bin

            cov_this_bin = pardict["outputfile"] + "_cov_" + format(z, ".2f") + ".txt"

            fisher = np.linalg.inv(np.loadtxt(cov_this_bin))

            fisher_main[b * 3 : (b + 1) * 3, b * 3 : (b + 1) * 3] += fisher[:3, :3]
            fisher_main[-num:, b * 3 : (b + 1) * 3] += fisher[-num:, :3]
            fisher_main[b * 3 : (b + 1) * 3, -num:] += fisher[:3, -num:]
            fisher_main[-num:, -num:] += fisher[-num:, -num:]

    return fisher_main
