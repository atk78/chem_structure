from typing import Literal
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm
from scipy.fftpack import fftn, ifftn
# import matplotlib.pyplot as plt
# from matplotlib import animation

from . import energy


class CahnHilliard:
    def __init__(
        self,
        c0: float,
        M: float,
        kappa: float,
        W: float,
        Nxy: int,
        Nz: int,
        dt: float = 0.1,
        dx: float = 1.0,
        noise: float = 0.1,
        n_steps: int = 1000,
        save_per_steps: int = 100,
        seed: int = 42
    ):
        self.c0 = c0
        self.M = M
        self.kappa = kappa
        self.W = W
        self.dt = dt
        self.Nxy = Nxy
        self.Nz = Nz
        self.dx = dx
        self.n_steps = n_steps
        self.save_per_steps = save_per_steps
        if self.Nz == 1:
            self.dim = 2
        elif self.Nz > 1:
            self.dim = 3
        else:
            raise ValueError("Nz must be greater than 0")
        self.save_dir = Path(__file__).parent.parent.joinpath("output").joinpath(
            f"c0_{c0}_M_{M}_kappa_{kappa}_W_{W}_dim_{self.dim}"
        )
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)
        self.npy_dir = self.save_dir.joinpath("npy")
        self.npy_dir.mkdir()

        np.random.seed(seed)

        if self.dim == 2:
            self.c = np.empty((self.n_steps + 1, Nxy, Nxy), dtype=np.float16)
            self.c_hat = np.zeros((Nxy, Nxy), dtype=np.complex64)
            self.dfdc_hat = np.zeros((Nxy, Nxy), dtype=np.complex64)
            kx = ky = np.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, indexing="ij"),
                dtype=np.float16
            )
        elif self.dim == 3:
            self.c = np.empty((self.n_steps + 1, Nxy, Nxy, Nz), dtype=np.float16)
            self.c_hat = np.zeros((Nxy, Nxy, Nz), dtype=np.complex64)
            self.dfdc_hat = np.zeros((Nxy, Nxy, Nz), dtype=np.complex64)
            kx = ky = np.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            kz = np.fft.fftfreq(Nz, d=dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, kz, indexing="ij"),
                dtype=np.float16
            )
        else:
            raise ValueError("dim must be 2 or 3")

        # self.L = self.N * self.dx
        self.c[0] = self.c0 + noise * np.random.standard_normal(self.c[0].shape)
        self.k_max_dealias = kx.max() * 2.0 / 3.0
        self.dealias = np.array(
            (np.abs(self.K[0]) < self.k_max_dealias)
            * (np.abs(self.K[1]) < self.k_max_dealias),
            dtype=bool
        )
        np.save(
            self.npy_dir.joinpath("0_step.npy"), self.c[0]
        )

    def compute_c(self):
        K2 = np.sum(self.K * self.K, axis=0, dtype=np.float16)
        self.c_hat[:] = fftn(self.c[0])
        for i in tqdm(range(1, self.n_steps + 1)):
            self.dfdc_hat[:] = fftn(energy.dfdc(self.c[i - 1], self.W))
            self.dfdc_hat *= self.dealias
            self.c_hat[:] = (
                (self.c_hat - self.dt * K2 * self.M * self.dfdc_hat)
                / (1 + self.dt * self.M * self.kappa * K2**2)
            )
            self.c[i] = ifftn(self.c_hat).real
            if i % self.save_per_steps == 0:
                np.save(
                    self.npy_dir.joinpath(f"{i}_step.npy"), self.c[i]
                )

# def init_field(
#     c0: float,
#     N: int,
#     dim: int = Literal[2, 3],
#     dx: float = 1.0,
#     noise: float = 0.1,
#     seed: int = 42
# ):
#     np.random.seed(seed)
#     if dim == 2:
#         c_hat = np.zeros((N, N), dtype=np.complex64)
#         dfdc_hat = np.zeros((N, N), dtype=np.complex64)
#         c = np.empty((N, N), dtype=np.float32)
#     elif dim == 3:
#         c_hat = np.zeros((N, N, N), dtype=np.complex64)
#         dfdc_hat = np.zeros((N, N, N), dtype=np.complex64)
#         c = np.empty((N, N, N), dtype=np.float32)
#     else:
#         raise ValueError("dim must be 2 or 3")
#     length = N * dx
#     c[0] = c0 + noise * np.random.standard_normal(c[0].shape)
#     return c, c_hat, dfdc_hat, length


# def init_condition(N: int, dx: int, dim: int = Literal[2, 3]):
#     # 2次元フーリエ変換のための波数ベクトルの設定
#     if dim == 2:
#         kx = ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
#         K = np.array(np.meshgrid(kx, ky, indexing="ij"), dtype=np.float32)
#     elif dim == 3:
#         kx = ky = kz = np.fft.fftfreq(N, d=dx) * 2 * np.pi
#         K = np.array(np.meshgrid(kx, ky, kz, indexing="ij"), dtype=np.float32)
#     else:
#         raise ValueError("dim must be 2 or 3")
#     K2 = np.sum(K * K, axis=0, dtype=np.float32)

#     k_max_dealias = kx.max() * 2.0 / 3.0
#     dealias = np.array(
#         (np.abs(K[0]) < k_max_dealias) * (np.abs(K[1]) < k_max_dealias),
#         dtype=bool
#     )
#     return K2, dealias


# def update_c_in_time(
#     c,
#     c_hat,
#     dfdc_hat,
#     M,
#     kappa,
#     W,
#     N,
#     dx,
#     dt,
#     dim,
#     n_steps,

# ):
#     K2, dealias = init_condition(N, dx, dim)
#     c_hat[:] = fftn(c[0])
#     for i in tqdm(range(1, n_steps)):
#         dfdc_hat[:] = fftn(energy.dfdc(c, W))
#         dfdc_hat *= dealias
#         c_hat[:] = (
#             (c_hat - dt * K2 * M * dfdc_hat)
#             / (1 + dt * M * kappa * K2**2)
#         )
#         c[i] = ifftn(c_hat).real
#     return c
