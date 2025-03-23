from pathlib import Path
import shutil
from typing import Optional, Literal

import numpy as np
import scipy.fftpack
from tqdm import tqdm
import scipy
import torch

from .. import img


class STrIPS:
    def __init__(
        self,
        M: float,
        kappa: float,
        W: float,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        dt: float = 1.0,
        output_dir: str = "output",
        device: Literal["cpu", "cuda"] = "cuda",
        random_seed: Optional[int] = None
    ):
        self.M = M  # mobility
        self.kappa = kappa  # gradient energy coefficient
        self.W = W  # double well potential coefficient
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        if device == "cuda":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def dfdc(self, c):
        return 2 * self.W * (c * (1 - c)**2 - (1 - c) * c**2)

    def bulk_free_energy(self, c):
        return self.W * c**2 * (1 - c)**2

    def calc_gradient(self, c):
        if isinstance(c, np.ndarray):
            if self.dim == 2:
                (gradient_x, gradient_y) = np.gradient(c, self.dx)
                return (gradient_x, gradient_y)
            else:
                (gradient_x, gradient_y, gradient_z) = np.gradient(c, self.dx)
                return (gradient_x, gradient_y, gradient_z)
        else:
            if self.dim == 2:
                (gradient_x, gradient_y) = torch.gradient(c, self.dx)
                return (gradient_x, gradient_y)
            else:
                (gradient_x, gradient_y, gradient_z) = torch.gradient(c, self.dx)
                return (gradient_x, gradient_y, gradient_z)

    def gradient_c2(self, c):
        gradient = self.calc_gradient(c)
        if isinstance(gradient[0], np.ndarray):
            return np.power(gradient, 2)
        else:
            return torch.pow(gradient, 2)

    def internal_free_energy(self, c):
        return 1 / 2 * self.kappa * self.gradient_c2(c)

    def free_energy(self, c):
        if isinstance(c, np.ndarray):
            F = (
                np.sum(self.bulk_free_energy(c))
                + np.sum(self.internal_free_energy(c))
            )
        else:
            F = (
                torch.sum(self.bulk_free_energy(c))
                + torch.sum(self.internal_free_energy(c))
            )
        return F

    def _setting_wave_number_vector(self):
        kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.Ny, d=self.dy) * 2 * np.pi
        kz = np.fft.fftfreq(self.Nz, d=self.dz) * 2 * np.pi
        kx_max_dealias = kx.max() * 2.0 / 3.0
        ky_max_dealias = ky.max() * 2.0 / 3.0
        kz_max_dealias = kz.max() * 2.0 / 3.0
        if self.dim == 2:
            K = np.array(
                np.meshgrid(ky, kx, indexing="ij"),
                dtype=np.float16
            )
            self.dealias = (
                (np.abs(K[0]) < ky_max_dealias)
                & (np.abs(K[1]) < kx_max_dealias)
            )
        else:
            K = np.array(
                np.meshgrid(ky, kx, kz, indexing="ij"),
                dtype=np.float16
            )
            self.dealias = (
                (np.abs(K[0]) < ky_max_dealias)
                & (np.abs(K[1]) < kx_max_dealias)
                & (np.abs(K[2]) < kz_max_dealias)
            )
        self.K2 = np.sum(K * K, axis=0, dtype=np.float16)
        self.c_hat = scipy.fftpack.fftn(self.c)

    def _convert_numpy_to_torch(self):
        self.c = torch.tensor(
            self.c, dtype=torch.float32, device=self.device
        )
        self.c_hat = torch.tensor(
            self.c_hat, dtype=torch.complex64, device=self.device
        )
        self.K2 = torch.tensor(
            self.K2, dtype=torch.float32, device=self.device
        )

    def _init_condition(self, c0: np.ndarray):
        self.c = c0.copy()
        if len(c0.shape) == 2:
            self.Ny, self.Nx = c0.shape
            self.Nz = 1
            self.dim = 2
        elif len(c0.shape) == 3:
            self.Ny, self.Nx, self.Nz = c0.shape
            self.dim = 3
        else:
            raise ValueError("The shape of c0 must be 2D or 3D.")
        self.porosity = (
            1 - c0.astype(np.float32).sum() / (self.Nx * self.Ny * self.Nz)
        )
        self.filename = (
            f"porosity_{self.porosity:.3f}"
            + f"_M_{self.M}"
            + f"_kappa_{self.kappa}"
            + f"_W_{self.W}"
            + f"_dim_{self.dim}"
        )
        self._make_dirs()
        self._setting_wave_number_vector()
        if self.device == "cuda":
            self._convert_numpy_to_torch()

    def _make_dirs(self):
        self.save_dir = self.output_dir.joinpath(self.filename)
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        self.save_dir.joinpath("npy").mkdir()
        self.save_dir.joinpath("tif").mkdir()
        self.save_dir.joinpath("img").mkdir()
        self.save_dir.joinpath("tmp").mkdir()

    def _calc_time_evolution(self):
        if self.device == "cuda":
            self.dfdc_hat = torch.fft.fftn(self.dfdc(self.c))
            self.dfdc_hat *= self.dealias
            self.c_hat = (
                (self.c_hat - self.dt * self.M * self.K2 * self.dfdc_hat)
                / (1 + self.dt * self.kappa * self.K2**2)
            )
            self.c = torch.fft.ifftn(self.c_hat).real
        else:
            self.dfdc_hat = scipy.fftpack.fftn(self.dfdc(self.c))
            self.dfdc_hat *= self.dealias
            self.c_hat = (
                (self.c_hat - self.dt * self.M * self.K2 * self.dfdc_hat)
                / (1 + self.dt * self.kappa * self.K2**2)
            )
            self.c = scipy.fftpack.ifftn(self.c_hat).real

    def calc_time_evolution(
        self, c0: np.ndarray,
        n_step: int,
        output_step: None | int,
        fps: int = 10,
    ):
        self._init_condition(c0)
        img.save_result(self.c, self.filename + "_0_step", self.save_dir)
        img.save_tmp_fig(
            self.c,
            self.filename + "_0_step",
            self.save_dir.joinpath("tmp"),
            self.dim
        )
        for step in tqdm(range(1, n_step + 1)):
            self._calc_time_evolution()
            if (output_step is not None) and (step % output_step == 0):
                img.save_result(
                    self.c,
                    self.filename + f"_{step}_step",
                    self.save_dir
                )
            if step % fps == 0:
                img.save_tmp_fig(
                    self.c,
                    self.filename + f"_{step}_step",
                    self.save_dir.joinpath("tmp"),
                    self.dim
                )
        img.save_result(self.c, self.filename + f"_{step}_step", self.save_dir)
        img.make_animation(
            self.save_dir.joinpath("img"),
            self.save_dir.joinpath("tmp")
        )


def generate_init_c(
    porosity: float,
    Nx: int,
    Ny: int,
    Nz: Optional[int] = None,
    noise: float = 0.1,
) -> np.ndarray:
    if (porosity < 0) or (porosity > 1):
        raise ValueError("Porosity must be between 0 and 1.")
    if (Nz is None) or (Nz == 1):
        c0 = (1 - porosity) + noise * np.random.standard_normal((Ny, Nx))
    else:
        c0 = (1 - porosity) + noise * np.random.standard_normal((Ny, Nx, Nz))
    return c0
