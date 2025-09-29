import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys

def main(path: str, show_fig: bool, title: str, ax: plt.axes, fiducial_values: list[int, int]) -> None:
    C = np.loadtxt(path)
    C = C[:2,:2]
    w0_fid, wa_fid = fiducial_values # fiducial values, change if needed

    # Eigen-decomposition for ellipse axes
    eigvals, eigvecs = np.linalg.eigh(C)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 1σ and 2σ scaling
    chi2_vals = [2.30, 6.17]
    colors = ["#1f77b4", "#ff7f0e"]
    labels = ["1σ contour", "2σ contour"]

    if ax is None:
        fig, ax = plt.subplots()

    for chi2, label, color in zip(chi2_vals, labels, colors):
        width, height = 2 * np.sqrt(eigvals * chi2)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        ell = Ellipse(
            xy=(w0_fid, wa_fid),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            lw=2,
            label=label,
        )
        ax.add_patch(ell)

    ax.axvline(w0_fid, color="k", ls="--", lw=1)
    ax.axhline(wa_fid, color="k", ls="--", lw=1)
    if True:
        ax.set_xlabel(r"$w_0$")
        ax.set_ylabel(r"$w_a$")
    if title is not None:
        ax.set_title(f"${title}$", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=10) 
    ax.legend(fontsize = 9)
    ax.add_patch(ell)
    ax.autoscale_view()
    if show_fig:
        plt.show()

if __name__ == "__main__":
    if "help" in sys.argv:
        print("plots confidence ellipses from output files from constrain_wowa.py. Supply path to output file.")
        sys.exit()
    try:
        path = sys.argv[1]
    except IndexError:
        raise IndexError("Failed to read path")
    fiducial_values = [-1.0, 0]
    if "--rerun" in sys.argv:
        print("Sorry, does not yet support rerunning automatically. Please manually rerun.")
        sys.exit()    
    if "--default" in sys.argv:
        print(f"default fiducial values are {fiducial_values}")
        sys.exit()

    main(path, fiducial_values=fiducial_values)