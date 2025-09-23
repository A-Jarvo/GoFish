import sys
from configobj import ConfigObj
import numpy as np
from ioutils import CosmoResults, InputData
import copy



def compute_cosmo_derivatives(
    parameter: str, pardict: ConfigObj, data: InputData
) -> list[float]:
    observables = ["h", "da", "fs8"] if include_fs8 else ["h", "da"]
    delta = 0.001
    pardict_up, pardict_down = copy.deepcopy(pardict), copy.deepcopy(pardict)
    pardict_up[parameter] = float(pardict_up[parameter]) + delta
    pardict_down[parameter] = float(pardict_down[parameter]) - delta
    cosmo_up = CosmoResults(pardict_up, data.zmin, data.zmax)
    cosmo_down = CosmoResults(pardict_down, data.zmin, data.zmax)

    if "fs8" in observables:
        cosmo_up.fs8 = cosmo_up.f * cosmo_up.sigma8
        cosmo_down.fs8 = cosmo_down.f * cosmo_down.sigma8

    derivatives = {
        obs: (np.array(getattr(cosmo_up, obs) - np.array(getattr(cosmo_down, obs))))
        / (2 * delta)
        for obs in observables
    }

    if False:
        print("cosmo_up.h:", cosmo_up.h)
        print("cosmo_down.h:", cosmo_down.h)
        print("cosmo_up.da:", cosmo_up.da)
        print("cosmo_down.da:", cosmo_down.da)
        if "fs8" in observables:
            print("cosmo_up.fs8:", cosmo_up.fs8)
            print("cosmo_down.fs8:", cosmo_down.fs8)

    return derivatives


def invert_matrix(matrix: np.array) -> np.array:
    try:
        # Attempt Cholesky decomposition
        L = np.linalg.cholesky(matrix)
        inverse_matrix = np.linalg.inv(L).T @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # Matrix is not positive definite, just pseudoinvert
        inverse_matrix = np.linalg.pinv(matrix)
    return inverse_matrix


def construct_Jacobians(
    derivatives: dict[str : dict[str:float]],
    parameters: list[str],
    redshift_bin_indices: list[int],
) -> np.array:
    observables = list(
        derivatives[next(iter(derivatives))].keys()
    )  # get keys of nested dict.
    # print(derivatives)

    Jacobian_element = lambda parameter, observable, redshift_index: derivatives[parameter][observable][redshift_index]
    Jacobian_Matrix = lambda redshift_index: np.array(
        [
            [
                Jacobian_element(parameter, observable, redshift_index)
                for parameter in parameters
            ]
            for observable in observables
        ]
    )

    Jacobians = [
        Jacobian_Matrix(redshift_index) for redshift_index in redshift_bin_indices
    ]
    return Jacobians


def main(configpath: str, include_fs8: bool):
    pardict = ConfigObj(configpath)

    # pardict initialisation copied from GoFish.py
    if "beta_phi_fixed" not in pardict:
        pardict["beta_phi_fixed"] = True
    if "geff_fixed" not in pardict:
        pardict["geff_fixed"] = True
    if "do_combined_DESI" not in pardict:
        pardict["do_combined_DESI"] = False
    if "pre_recon" not in pardict:
        pardict["pre_recon"] = False

    if not pardict.as_bool("beta_phi_fixed") and not pardict.as_bool("BAO_only"):
        raise ValueError(
            "You have set beta_phi_= False and BAO_only = False. This is not allowed."
        )

    if not pardict.as_bool("geff_fixed") and not pardict.as_bool("BAO_only"):
        raise ValueError(
            "You have set geff_fixed = False and BAO_only = False. This is not allowed."
        )

    if "outputfile" not in pardict:
        raise ValueError("No 'outputfile' specified in supplied config.")

    output_path = pardict["outputfile"]

    # Read in the file containing the redshift bins, nz and bias values
    data = InputData(pardict)

    # easier to use later like this.
    redshift_bins = CosmoResults(pardict, data.zmin, data.zmax).z
    redshift_bin_indices = list(range(0, len(redshift_bins)))

    covariance_matrices_obs = []
    missing_cov_files = []
    for redshift_bin in redshift_bins:
        try:
            cov_matrix = np.loadtxt(
                output_path + f"_cov_{format(redshift_bin, '.2f')}.txt", dtype=float
            )  # import to np array
            covariance_matrices_obs.append(
                cov_matrix[-2 - int(include_fs8) :, -2 - int(include_fs8) :]
            )  # append only H, Da sub matrix
        except FileNotFoundError: # for some reason no cov matrix for redshift
            missing_cov_files.append(redshift_bin)
            
    for redshift in missing_cov_files:
        index = np.where(redshift_bins == redshift)
        np.delete(redshift_bins, index)
        redshift_bin_indices.pop()

    parameters_to_constrain = ["w0_fld", "wa_fld", "omega_cdm", "omega_b", "h"]  # theoretically can work for any

    derivatives = {
        parameter: compute_cosmo_derivatives(parameter, pardict, data)
        for parameter in parameters_to_constrain
    }
    # del pardict, data, h, Da

    Jacobians = construct_Jacobians(
        derivatives, parameters_to_constrain, redshift_bin_indices
    )

    # print(Jacobian)

    fisher_matrices_obs = [invert_matrix(cov) for cov in covariance_matrices_obs]

    for index in redshift_bin_indices:
        np.savetxt(output_path + f"_HDa_cov_{format(redshift_bins[index], '.2f')}_.txt", fisher_matrices_obs[index])
    
    fisher_matrices_para = [
        Jacobians[z_index].T @ fisher_matrices_obs[z_index] @ Jacobians[z_index]
        for z_index in redshift_bin_indices
    ]
    total_cov_para = invert_matrix(sum(fisher_matrices_para))
    cov_matrices_para = [invert_matrix(fisher) for fisher in fisher_matrices_para]

    for index in redshift_bin_indices:
        file_name = output_path + f"_w0wa_cov_{format(redshift_bins[index], '.2f')}.txt"
        np.savetxt(
            file_name, cov_matrices_para[index]
        )  # output individual cov matrices
    np.savetxt(
        output_path + "_w0wa_cov_full.txt", total_cov_para
    )  # output full cov matrix

    print("Overall covaraince matrix:")
    print(total_cov_para)
    print("w0wa coraviance matrix:")
    print(total_cov_para[0:2,0:2])
    for param,index in zip(parameters_to_constrain,range(0,len(parameters_to_constrain))):
        print(
        f"1sigma error for {param}: {np.sqrt(total_cov_para[index][index])}"
        )

    if False:
        print("derivatives['h']:", derivatives["w0_fld"]["h"][:5])
        print("derivatives['da']:", derivatives["w0_fld"]["da"][:5])
        print("covariance matrix (H,Da):", covariance_matrices_obs[0])
        print("Fisher matrix obs:", fisher_matrices_obs[0])
        print("Fisher matrix params:", fisher_matrices_para[0])
        print("F_obs shape:", fisher_matrices_obs[0].shape)  # should be 3x3
        print("Jacobian shape:", Jacobians[0].shape)  # should be 3x2
        print("F_params shape:", fisher_matrices_para[0].shape)  # should be 2x2
        print("derivatives['fs8']:", derivatives["w0_fld"]["fs8"][:5])


if __name__ == "__main__":
    try:
        configpath: str = sys.argv[1]
    except IndexError:
        raise TypeError("Please supply a config file.")

    if "help" in sys.argv:
        print(
            "Computes covariance matrix for parameters (w0,wa). Supply path to config file as first parameter."
            "optionally supply '--rerun' to recompute GoFish.py, else will attempt to draw from output specified in config."
        )
        sys.exit()
    if "--rerun" in sys.argv:
        import subprocess
        print("Rerunning GoFish.py")
        subprocess.run([sys.executable, "src/GoFish.py", sys.argv[1]], check=True)
    include_fs8 = "--include_fs8" in sys.argv
    # include_fs8 = True
    main(sys.argv[1], include_fs8)
