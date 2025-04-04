from pathlib import Path
import shutil

import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from PIL import Image


def save_result(c: np.ndarray, filename: str, output_dir: Path):
    if isinstance(c, torch.Tensor):
        c = c.cpu().detach().numpy()
    c = (c - c.min()) / (c.max() - c.min())
    c = c.astype(np.float16)
    tif_c = np.where(c < 0.5, 0, 255)
    tif_c = tif_c.astype(np.uint8)
    np.save(
        output_dir.joinpath("npy").joinpath(f"{filename}.npy"), c
    )
    if len(c.shape) == 2:
        tifffile.imwrite(
            output_dir.joinpath("tif").joinpath(f"{filename}.tif"),
            tif_c
        )
        fig = fig_2d(c, filename)
    elif len(c.shape) == 3:
        tifffile.imwrite(
            output_dir.joinpath("tif").joinpath(f"{filename}.tif"),
            tif_c.transpose((2, 0, 1))
        )
        fig = fig_3d(c, filename)
    fig.savefig(
        output_dir.joinpath("img").joinpath(f"{filename}.png")
    )
    plt.close(fig)


def save_tmp_fig(c: np.ndarray | torch.Tensor, filename: str, tmp_dir: Path, dim: int):
    if (c.min() < 0) or (c.max() > 1):
        c = (c - c.min()) / (c.max() - c.min())
    c = c.astype(np.float16)
    if dim == 2:
        tmp_fig = fig_2d(c, filename)
    elif dim == 3:
        tmp_fig = fig_3d(c, filename)
    tmp_fig.savefig(tmp_dir.joinpath(f"{filename}.png"))
    tmp_fig.clear()
    plt.close(tmp_fig)  # メモリ解放


def fig_3d(c: np.ndarray, title: str):
    X, Y, Z = np.meshgrid(
        np.arange(c.shape[1]),
        np.arange(c.shape[0]),
        c.shape[2] - np.arange(c.shape[2])
    )

    kw = {
        "vmin": 0,
        "vmax": 1,
        "levels": np.linspace(0, 1, 21),
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
    ax.set_title(title)

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label="c")
    return fig


def fig_2d(c: np.ndarray, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(c, cmap="jet", vmin=0, vmax=1)
    ax.set_xlabel("X [mesh]")
    ax.set_ylabel("Y [mesh]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("c")
    ax.set_title(title)
    return fig


def make_animation(img_dir: Path, tmp_dir:  Path):
    imgs = [Image.open(str(png)) for png in tmp_dir.glob("*.png")]
    imgs[0].save(
        fp=img_dir.joinpath("anime.gif"),
        save_all=True,  # gif形式で保存する
        append_images=imgs[1:],  # 後続の残りの画像
        optimize=True,  # たまにおかしくなる。その時はFalse
        duration=100,  # フレーム時間 ms
        loop=0  # loopを行う
    )
    shutil.rmtree(tmp_dir)
