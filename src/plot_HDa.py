# src/make_plot.py
import sys
import matplotlib.pyplot as plt
from configobj import ConfigObj
from ioutils import CosmoResults, InputData

def main(config_path: str) -> None:
    config = ConfigObj(config_path)
    data = InputData(config)
    cosmo = CosmoResults(config, data.zmin, data.zmax)

    plt.plot(cosmo.z, cosmo.da, label="D_A(z)")
    plt.plot(cosmo.z, cosmo.h, label="H(z)")
    plt.xlabel("Redshift z")
    plt.ylabel("Distance / Hubble Rate")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    try:
        config_path: str = sys.argv[1]
    except IndexError:
        raise TypeError("Please supply a config file.")
    if ("help" in sys.argv) or ("--help" in sys.argv):
        print("Produces a plot of H and Da vs redshift using output data from GoFish. Supply config path after calling function. " \
        "Supply help or --help to view this message again. Supply --rerun to rerun GoFish with supplied config")
        sys.exit()
    elif "--rerun" in sys.argv:
        import subprocess
        print("Rerunning GoFish.py")
        subprocess.run([sys.executable, "src/GoFish.py", config_path], check=True)
    main(config_path)
