import pandas as pd
from scipy.interpolate import PchipInterpolator
import sys
from configobj import ConfigObj
from ioutils import CosmoResults
import pprint

def main(main_config_path: str = None, pardict: ConfigObj = None, 
         print_file: bool = False, overrite_input_file_cosmo: str = None):
    if main_config_path is not None:
        pardict = ConfigObj(main_config_path)
    input_file = pardict["inputfile"]
    output_file = pardict["outputfile"]
    new_z_min = float(pardict["zmin"])
    new_z_max = float(pardict["zmax"])
    bin_width = float(pardict["bin_width"])
    cosmo_config_path = pardict["cosmo_config"]
    cosmo_pardict = ConfigObj(cosmo_config_path)
    old_data = pd.read_csv(
        input_file,
        delim_whitespace=True,
        dtype="float",
        skiprows=0,
        escapechar="#",
    )
    if overrite_input_file_cosmo:
        cosmo_pardict["inputfile"] = overrite_input_file_cosmo

    new_z_bins = get_new_z(new_z_min, new_z_max, bin_width)
    tracer_cols, bias_cols = get_tracer_col_names(old_data)
    new_data = pd.DataFrame(new_z_bins, columns=[" zmin", "zmax"])

    for tracer_col, bias_col in zip(tracer_cols, bias_cols):
        rebin_tracer(old_data, new_data, new_z_bins, tracer_col, cosmo_pardict)
        bias = old_data.at[0, bias_col]
        new_data[bias_col] = bias
        new_data.to_csv(f'{output_file}', sep='\t', index=False, header=True)
    if print_file:
        print(new_data)


def get_new_z(zmin: float, zmax: float, bin_width: float) -> list[tuple[float, float]]:
    prev_z = zmin
    curr_z = prev_z
    new_z_bins = []
    assert zmax > zmin
    assert zmax > zmin+bin_width
    while curr_z < zmax:
        curr_z += bin_width
        new_z_bin = (prev_z, curr_z)
        new_z_bins.append(new_z_bin)
        prev_z = curr_z
    new_z_bins = [(round(low,3), round(high,3)) for low, high in new_z_bins]# round
    return new_z_bins

def get_tracer_col_names(data: pd.DataFrame) -> tuple[tuple[str], tuple[str]]:
    tracer_col_names = []
    bias_col_names = []
    for col in data.columns:
        if "bias" in col:
            bias_col_names.append(col)
        elif "_" in col:
            tracer_col_names.append(col)
    
    tracer_col_names = tuple(tracer_col_names)
    bias_col_names = tuple(bias_col_names)
    return tracer_col_names, bias_col_names

def rebin_tracer(old_data: pd.DataFrame, new_data: pd.DataFrame, z_bins: list[tuple[float, float]],
                 tracer_col: str, cosmo_pardict: ConfigObj):
    interpolating_poly = interpolate(old_data, tracer_col, cosmo_pardict)
    zmin, zmax = zip(*z_bins)
    zmin = pd.Series(zmin)
    zmax = pd.Series(zmax)
    new_cosmo = CosmoResults(cosmo_pardict, zmin, zmax)
    new_volumes = new_cosmo.volume
    get_tracer_count = lambda zmin, zmax, volume: (interpolating_poly(zmax) - interpolating_poly(zmin)) / volume
    for index, (z_bin, volume) in enumerate(zip(z_bins, new_volumes)):
        tracer_count = get_tracer_count(*z_bin, volume)
        new_data.at[index, tracer_col] = round(tracer_count)
        

def interpolate(data: pd.DataFrame, col_name: str, cosmo_pardict: ConfigObj):
    z_min = data[" zmin"]
    z_max = data["zmax"]
    z_bins = list(zip(z_min, z_max))
    cosmo = CosmoResults(cosmo_pardict, z_min, z_max)
    volumes = cosmo.volume
    tracer_counts = [density*volume for density, volume in zip(data[col_name], volumes)]
    tracer_count_cumulative = [0]
    tracer_count_cumulative.append(tracer_counts[0])
    for tracer_count in tracer_counts[1:]:
        tracer_count_cumulative.append(tracer_count_cumulative[-1]+tracer_count)
    z_max = pd.concat([pd.Series([0]), z_max])
    interpolator = PchipInterpolator(z_max, tracer_count_cumulative)
    return interpolator
    

def defaults() -> ConfigObj:
    raise NotImplemented("no defaults implemented")

if __name__ == '__main__':
    overrite_input_file_cosmo = None
    printfile = False
    if "--overrite_input_file" in sys.argv:
        flag_index = sys.argv.index("--overrite_input_file")
        try:
            overrite_input_file_cosmo = sys.argv[+1]
            assert isinstance(overrite_input_file_cosmo, str)
        except (IndexError, AssertionError):
            print("failed reading input file, using default")
            overrite_input_file_cosmo = "input_files/DESI_All_nbar_validationpaper.txt"
    if "help" in sys.argv:
        help_statement = "Sorry, 'help' is not yet supported"
        print(help_statement)
        sys.exit()
    if "--defaults" in sys.argv:
        print("printing default main config")
        pprint.pp(defaults())
    if "--printfile" in sys.argv:
        printfile = True
    try:
        main_config_path = sys.argv[1]
        assert type(main_config_path) == str
    except (IndexError, AssertionError):
        print("failed to read config path, using default")
        pardict = defaults()
        main(pardict=pardict)
    else:
        main(main_config_path=main_config_path, print_file=printfile,
             overrite_input_file_cosmo=overrite_input_file_cosmo)