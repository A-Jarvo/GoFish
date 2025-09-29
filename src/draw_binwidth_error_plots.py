import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import math
from copy import deepcopy



def main(pardict: dict, path: str, cosmologies: list[str]) -> None: # absolutely horroundes code. Wrote while real tired
    cols_to_plot = pardict["cols_to_plot"]
    plots_restrict = pardict["restrict_plots_to_index_in_legend"]
    if plots_restrict is None:
        pardict["title"] = pardict["title"][0]
    else:
        cosmo = pardict["legend"][plots_restrict]
        pardict["title"] = pardict["title"][1].format(cosmo)
        cols_to_plot = cols_to_plot[plots_restrict]
    
    x_to_plot = pardict["x_to_plot"]

    data = read_input(path, cosmologies)
    print("Data table:")
    print(data)

    if isinstance(cols_to_plot, list): # multiple plots on one fig
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for col in cols_to_plot:
            draw_subplot(ax, data, x_to_plot, col)
        decorate_subplot(pardict, ax)
        ax.legend(pardict["legend"], fontsize=12)
    else: # single plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        col = cols_to_plot
        print(cols_to_plot)
        draw_subplot(ax, data, x_to_plot, col)
        decorate_subplot(pardict, ax)
    
    plt.show()

def draw_subplot(ax: plt.axes, data: pd.DataFrame, x_col_head: str, y_col_head: str) -> None:
    x_col = data[x_col_head]
    print(y_col_head)
    y_col = data[y_col_head]
    ax.plot(x_col, y_col, "o--", ms=4.5)

def decorate_subplot(pardict: dict, ax: plt.axes, **formatting) -> None:
    title = pardict["title"]
    xlabel = pardict["xlabel"]
    ylabel = pardict["ylabel"]
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def read_input(path: str, cosmologies: str) -> pd.DataFrame:
    col_index = pardict["col_index"]
    sigma = pardict["sigma"]
    recompute_errors_override = pardict["recompute_errors_override"]

    data = pd.read_csv(path, index_col=col_index)
    data[col_index] = data.index

    if sigma:
        chi2 = math.erf(sigma/np.sqrt(2))
    for cosmo in cosmologies:
        if recompute_errors_override:
            data[f"{cosmo}_sigma"] = np.pi * chi2 * np.sqrt((data[f"{cosmo}_sigma_w0"] * data[f"{cosmo}_sigma_wa"])**2 - data[f"{cosmo}_cov"]**2)
            continue
        if f"{cosmo}_sigma" in data.columns:
            continue
        data[f"{cosmo}_sigma"] = np.pi * chi2 * np.sqrt((data[f"{cosmo}_sigma_w0"] * data[f"{cosmo}_sigma_wa"])**2 - data[f"{cosmo}_cov"]**2)
    data["bin_count"] = np.ceil(2.1 / data.index)

    return data

def default_config() -> dict[str: ]:
    pardict = {}
    pardict["path"] = "output_files/binning_csv/binwidth_data.csv"
    pardict["recompute_errors_override"] = False
    pardict["sigma"] = 1
    pardict["cols_to_plot"] = ["lambda_sigma", "w0wa_sigma"]
    pardict["x_to_plot"] = "bin_width"
    pardict["col_index"] = "bin_width"
    pardict["title"] = ["Figure 1: Plot of Area of Error Ellipse as a Function \n of Bin Width for Fiducial Cosmologies ΛCDM and w0waCDM",
                        "Figure 1: Plot of Area of Error Ellipse as a Function \n of Bin Width for Fiducial Cosmology {}"]
    pardict["xlabel"] = "Bin Width"
    pardict["ylabel"] = f"Area of {pardict["sigma"]}σ Error Ellipse"
    pardict["legend"] = ["ΛCDM", "w0waCDM"]
    pardict["restrict_plots_to_index_in_legend"] = None
    return pardict

if __name__ == "__main__":
    pardict = default_config()
    try:
        path = sys.argv[1]
    except IndexError:
        "No supplied path, using default."
        path = pardict["path"]
    if "--help" in sys.argv:
        help_message = "sorry, '--help' is not yet supported"
        print(help_message)
        exit()
    if "--defaults" in sys.argv:
        for key, val in pardict.items():
            print(f"{key} = {val}")
        exit()
    cosmologies = ["lambda", "w0wa"]
    main(pardict, path, cosmologies)