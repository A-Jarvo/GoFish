import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
import sys
from configobj import ConfigObj
from math import ceil

def main(config_path)-> None:

    pardict = ConfigObj(config_path)
    
    old_path = pardict["inputfile"]
    new_path = pardict["outputfile"]
    old_data = pd.read_csv(
            old_path,
            delim_whitespace=True,
            dtype="float",
            skiprows=0,
            escapechar="#",
        )
    new_z_low, new_z_high = get_z(pardict, old_data["zmax"].max())
    new_data = pd.DataFrame({"# zmin": new_z_low, "zmax": new_z_high})
    tracer_col_names, bias_col_names = get_tracers(old_data)
    for tracer_col_name, bias_col_name in zip(tracer_col_names, bias_col_names):
        new_tracer_data = rebin_tracer(old_data, zip(new_z_low, new_z_high), tracer_col_name)
        new_data[tracer_col_name] = new_tracer_data
        new_data[bias_col_name] = old_data[bias_col_name].iloc[0]
    new_data.to_csv(f'{new_path}', sep='\t', index=False, header=True)
    print(new_data)
    sys.exit()
    
def rebin_tracer(dataframe, new_z_bins, tracer_col_name):
    new_tracer_data = []
    for new_z_bin in new_z_bins:
        total_tracer_data_in_bin = 0
        for old_index in dataframe.index:
            old_bin_dataframe = dataframe.iloc[old_index]
            old_z_bin = (old_bin_dataframe[" zmin"], old_bin_dataframe["zmax"]) # get in same form as new bins
            overlap = calculate_overlap(old_z_bin, new_z_bin)
            if overlap > 0:
                total_tracer_data_in_bin += overlap * old_bin_dataframe[tracer_col_name]
        new_tracer_data.append(total_tracer_data_in_bin)
    return new_tracer_data


def calculate_overlap(old_interval, new_interval):
    old_start, old_end = old_interval
    new_start, new_end = new_interval

    overlap_start = max(old_start, new_start)
    overlap_end = min(old_end, new_end)
    overlap_width = max(0, overlap_end - overlap_start)
    old_width = old_end - old_start
    return overlap_width / old_width
    



def get_z(pardict, old_z_max) -> tuple[list[float], list[float], list[float]]:
    # not properly implemented, just harded coded rn
    bin_width = float(pardict["bin_width"])
    if ("zmin" in pardict) and ("zmax" in pardict):
        z_min = float(pardict["zmin"])
        z_max = float(pardict["zmax"])
    else:
        z_min = 0
        z_max = bin_width * ceil(old_z_max / bin_width)
    
    assert z_max % bin_width

    all_z =  [z / 100 for z in range(int(z_min*100), int(z_max*100)+1, int(bin_width*100))]
    z_low = all_z[0:-1]
    z_high = all_z[1:]

    return z_low, z_high


def read_input(path: str)-> tuple[pd.DataFrame, list[float]]:
    data = pd.read_csv(
            path,
            delim_whitespace=True,
            dtype="float",
            skiprows=0,
            escapechar="#",
        )
    tracer_col_names, bias_col_names = get_tracers(data)

    #assert data[bias_col_names] == data[bias_col_names].iloc[0].all() # check biases are constant
    biases = data[bias_col_names].iloc[0].tolist

    tracer_data = data[tracer_col_names]

    z_mid = (data[" zmin"]+ data["zmax"])/2
    z_range = data["zmax"].iloc[0] - data[" zmin"].iloc[0]

    return z_mid, z_range, tracer_data, biases


def get_tracers(dataframe: pd.DataFrame) -> tuple[list[str], list[float]]:
    # takes in dataframe and returns a list of column names
    # with galaxy data and a list of column names with bias data
    cols = list(dataframe.keys())
    cols = [col for col in cols if col not in [col for col in [" zmin", "zmax", "volume"] if col in cols]] # subtract non galaxy cols
    nbar_or_nz = f"{"nz" if any("nz" in col for col in cols) else "nbar"}"
    tracer_col_names = [col for col in cols if nbar_or_nz in col]
    bias_col_names = [col for col in cols if "bias" in col]
    return tracer_col_names, bias_col_names
    

if __name__ == '__main__':
    if "help" in sys.argv:
        help_statement = "Sorry, 'help' is not yet supported"
        print(help_statement)
        sys.exit()
    try:
        config_path = sys.argv[1]
        assert type(config_path) == str
    except KeyError or AssertionError:
        print("please supply the path to the config file as a string")
        sys.exit()
    main(config_path)