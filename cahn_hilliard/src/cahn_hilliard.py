import math
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm
from scipy.fftpack import fftn, ifftn

from . import energy, record


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
        output_dir: str = "output",
        seed: int = 42
    ):
        self.M = M
        self.kappa = kappa
        self.W = W
        self.dt = dt
        self.Nxy = Nxy
        self.Nz = Nz
        self.dx = dx
        self.n_steps = n_steps
        self.save_per_steps = save_per_steps
        self.cycle = math.ceil(n_steps / save_per_steps)
        if self.Nz == 1:
            self.dim = 2
        elif self.Nz > 1:
            self.dim = 3
        else:
            raise ValueError("Nz must be greater than 0")

        np.random.seed(seed)

        if self.dim == 2:
            self.init_c = c0 + noise * np.random.standard_normal((Nxy, Nxy))
            self.c_hat = np.zeros((Nxy, Nxy), dtype=np.complex64)
            self.dfdc_hat = np.zeros((Nxy, Nxy), dtype=np.complex64)
            kx = ky = np.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, indexing="ij"),
                dtype=np.float16
            )
            self.c0 = self.init_c.sum() / (Nxy**2)
        elif self.dim == 3:
            self.init_c = c0 + noise * np.random.standard_normal((Nxy, Nxy, Nz))
            self.c_hat = np.zeros((Nxy, Nxy, Nz), dtype=np.complex64)
            self.dfdc_hat = np.zeros((Nxy, Nxy, Nz), dtype=np.complex64)
            kx = ky = np.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            kz = np.fft.fftfreq(Nz, d=dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, kz, indexing="ij"),
                dtype=np.float16
            )
            self.c0 = self.init_c.sum() / (Nxy**2 * Nz)
        else:
            raise ValueError("dim must be 2 or 3")

        self.init_c = self.init_c.astype(np.float16)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.save_dir = self.output_dir.joinpath(
            f"c0_{self.c0:.3f}_M_{M}_kappa_{kappa}_W_{W}_dim_{self.dim}"
        )
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir()
        self.save_dir.joinpath("npy").mkdir()
        self.save_dir.joinpath("img").mkdir()
        self.save_dir.joinpath("tif").mkdir()

        # self.L = self.N * self.dx

        self.k_max_dealias = kx.max() * 2.0 / 3.0
        self.dealias = np.array(
            (np.abs(self.K[0]) < self.k_max_dealias)
            * (np.abs(self.K[1]) < self.k_max_dealias),
            dtype=bool
        )
        record.save_result(self.save_dir, self.init_c, "0_step")

    def compute_c(self):
        K2 = np.sum(self.K * self.K, axis=0, dtype=np.float16)
        self.c_hat[:] = fftn(self.init_c)
        step = 0
        for cycle in tqdm(range(self.cycle)):
            if self.dim == 2:
                self.c = np.empty(
                    (self.save_per_steps + 1, self.Nxy, self.Nxy),
                    dtype=np.float16
                )
            if self.dim == 3:
                self.c = np.empty(
                    (self.save_per_steps + 1, self.Nxy, self.Nxy, self.Nz),
                    dtype=np.float16
                )
            self.c[0] = self.init_c
            for i in range(1, self.save_per_steps + 1):
                self.dfdc_hat[:] = fftn(energy.dfdc(self.c[i - 1], self.W))
                self.dfdc_hat *= self.dealias
                self.c_hat[:] = (
                    (self.c_hat - self.dt * K2 * self.M * self.dfdc_hat)
                    / (1 + self.dt * self.M * self.kappa * K2**2)
                )
                self.c[i] = ifftn(self.c_hat).real
                step += 1
                if step > self.n_steps:
                    break
                if i == self.save_per_steps:
                    self.init_c = self.c[i]
                    record.save_result(self.save_dir, self.c[i], step)


