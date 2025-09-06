import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys

def main() -> None:
    C = np.loadtxt(path)
    w0_fid, wa_fid = -1.0, 0.0 # fiducial values, change if needed

    # Eigen-decomposition for ellipse axes
    eigvals, eigvecs = np.linalg.eigh(C)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 1σ and 2σ scaling
    chi2_vals = [2.30, 6.17]
    colors = ["#1f77b4", "#ff7f0e"]
    labels = ["1σ contour", "2σ contour"]

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
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_a$")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError:
        raise TypeError("Supply path to covariance matrix.")
    if "help" in sys.argv:
        print("plots confidence ellipses from output files from constrain_wowa.py. Supply path to output file.")
        sys.exit()
    if "--rerun" in sys.argv:
        print("Sorry, does not yet support rerunning automatically. Pleas manually rerun.")
        sys.exit
    main(path)