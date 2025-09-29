import draw_ellipses #main(path, False)
import matplotlib.pyplot as plt

def main(paths: list[str], titles: list[str], fiducial_values: list[int, int], *args, **kwargs):
    if len(args) != 0:
        raise KeyError("*args must be empty")
    set_axis_equal = False
    if "set_axis_equal" in kwargs.keys():
        set_axis_equal = kwargs["set_axis_equal"]
    num_plots = len(titles)
    if "subplot_dimensions" in kwargs.keys():
        nrows, ncols = kwargs["subplot_dimensions"]
    elif "cosmologies" in kwargs.keys():
        nrows = len(kwargs["cosmologies"])
        ncols = int(num_plots / nrows)
    elif "bin_widths" in kwargs.keys():
        ncols = len(kwargs["bin_widths"])
        nrows = int(num_plots / ncols)
    else:
        print("failed to read full subplot dimensions, using default. May lead to unexpected results.")
        nrows, ncols = [2, 2]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True)
    past_ax = None
    for ax, path, title, fiducial_value in zip(axs.flatten(), paths, titles, fiducial_values):
        assert ax != past_ax
        draw_ellipses.main(path, False, title=title, ax=ax, fiducial_values=fiducial_value)
        past_ax = ax

    if set_axis_equal:
        x_lim = [0, 0]
        y_lim = [0, 0]
        axes_width = lambda axes_lims: axes_lims[1] - axes_lims[0]
        for ax in axs.flatten():
            if axes_width(ax.get_xlim()) > axes_width(x_lim):
                x_lim = ax.get_xlim()
            if axes_width(ax.get_ylim()) > axes_width(y_lim):
                y_lim = ax.get_ylim()
        print(x_lim, y_lim, sep=" ")
        for ax in axs.flatten():
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    plt.show()

    
    
if __name__ == "__main__":
    bin_widths = [0.1, 0.44]
    cosmologies = ["ΛCDM", r"w_0w_aCDM"]
    cosmologies_for_path = ["LambdaCDM", "w0waCDM"]
    titles = [f"{cosmo} - {bin_width}"
              for bin_width in bin_widths
              for cosmo in cosmologies]
    paths = [f"output_files/GoFish_DESI_{cosmo}_w0wa_cov_full_{bin_width}.txt" 
             for bin_width in bin_widths 
             for cosmo in cosmologies_for_path]
    fiducial_values = [[-1, 0], [-0.762, -0.81]] * len(bin_widths)
    print(fiducial_values)
    main(paths, titles, cosmologies=cosmologies, fiducial_values=fiducial_values,
         bin_widths=bin_widths, set_axis_equal=False,
         subplot_dimensions = [len(cosmologies), len(bin_widths)])