from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt


def save_result(
    save_dir: Path,
    c: np.ndarray,
    step: str
):
    c = (c - c.min()) / (c.max() - c.min())
    c = c.astype(np.float16)
    np.save(
        save_dir.joinpath("npy").joinpath(f"{step}_step.npy"),
        c
    )
    if len(c.shape) == 2:
        tifffile.imwrite(
            save_dir.joinpath("tif").joinpath(f"{step}_step.tif"),
            c
        )
        fig = fig_2d(c, step)
    elif len(c.shape) == 3:
        tifffile.imwrite(
            save_dir.joinpath("tif").joinpath(f"{step}_step.tif"),
            c.transpose((2, 0, 1))
        )
        fig = fig_3d(c, step)
    fig.savefig(
        save_dir.joinpath("img").joinpath(f"{step}_step.png")
    )
    plt.close(fig)


def fig_3d(c: np.ndarray, step):
    X, Y, Z = np.meshgrid(
        np.arange(c.shape[0]),
        np.arange(c.shape[1]),
        c.shape[2] - np.arange(c.shape[2])
    )

    kw = {
        "vmin": c.min(),
        "vmax": c.max(),
        "levels": np.linspace(c.min(), c.max(), 21),
        "cmap": "jet"
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], c[:, :, 0],
        zdir="z", offset=Z.max(), **kw
    )
    _ = ax.contourf(
        X[0, :, :], c[0, :, :], Z[0, :, :],
        zdir="y", offset=0, **kw
    )
    C = ax.contourf(
        c[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir="x", offset=X.max(), **kw
    )

    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color="0.4", linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel="X [mesh]",
        ylabel="Y [mesh]",
        zlabel="Z [mesh]",
    )

    # Set zoom and angle view
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=0.9)
    ax.set_title(f"{step} step")

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label="c")
    return fig


def fig_2d(c: np.ndarray, step: int):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(c, cmap="jet")
    ax.set_xlabel("X [mesh]")
    ax.set_ylabel("Y [mesh]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("c")
    ax.set_title(f"{step} step")
    return fig
