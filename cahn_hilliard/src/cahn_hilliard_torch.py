import math
from pathlib import Path
import shutil

import numpy as np
from tqdm import tqdm
import torch
# from scipy.fftpack import fftn, ifftn
# import matplotlib.pyplot as plt
# from matplotlib import animation

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
        seed: int = 42,
        device: str = "cuda"
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
        if (torch.cuda.is_available()) and (device == "cuda"):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.dim == 2:
            self.init_c = c0 + torch.tensor(
                noise * np.random.standard_normal((Nxy, Nxy)),
                device=self.device
            )
            self.c_hat = torch.zeros(
                (Nxy, Nxy),
                dtype=torch.complex64,
                device=self.device
            )
            self.dfdc_hat = torch.zeros(
                (Nxy, Nxy),
                dtype=torch.complex64,
                device=self.device
            )
            kx = ky = torch.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            kx, ky = torch.meshgrid(kx, ky, indexing="ij")
            self.K = torch.stack((kx, ky)).to(self.device)
            self.c0 = self.init_c.cpu().numpy().sum() / (Nxy**2)
        elif self.dim == 3:
            self.init_c = c0 + torch.tensor(
                noise * np.random.standard_normal((Nxy, Nxy, Nz)),
                device=self.device
            )
            self.c_hat = torch.zeros(
                (Nxy, Nxy, Nz),
                dtype=torch.complex64,
                device=self.device
            )
            self.dfdc_hat = torch.zeros(
                (Nxy, Nxy, Nz),
                dtype=torch.complex64,
                device=self.device
            )
            kx = ky = torch.fft.fftfreq(Nxy, d=dx) * 2 * np.pi
            kz = torch.fft.fftfreq(Nz, d=dx) * 2 * np.pi
            kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")
            self.K = torch.stack((kx, ky, kz)).to(self.device)
            self.c0 = self.init_c.cpu().numpy().sum() / (Nxy**2 * Nz)
        else:
            raise ValueError("dim must be 2 or 3")

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
        self.dealias = (torch.abs(self.K[0]) < self.k_max_dealias) * (torch.abs(self.K[1]) < self.k_max_dealias)
        record.save_result(self.save_dir, self.init_c.cpu().numpy(), "0_step")

    def compute_c(self):
        K2 = torch.sum(self.K * self.K, axis=0, dtype=torch.float16)
        self.c_hat[:] = torch.fft.fftn(self.init_c)
        step = 0
        for cycle in tqdm(range(self.cycle)):
            if self.dim == 2:
                self.c = torch.empty(
                    (self.save_per_steps + 1, self.Nxy, self.Nxy),
                    dtype=torch.float32,
                    device=self.device
                )
            if self.dim == 3:
                self.c = torch.empty(
                    (self.save_per_steps + 1, self.Nxy, self.Nxy, self.Nz),
                    dtype=torch.float32,
                    device=self.device
                )
            self.c[0] = self.init_c
            for i in range(1, self.save_per_steps + 1):
                self.dfdc_hat[:] = torch.fft.fftn(energy.dfdc(self.c[i - 1], self.W))
                self.dfdc_hat *= self.dealias
                self.c_hat[:] = (
                    (self.c_hat - self.dt * K2 * self.M * self.dfdc_hat)
                    / (1 + self.dt * self.M * self.kappa * K2**2)
                )
                self.c[i] = torch.fft.ifftn(self.c_hat).real
                step += 1
                if step > self.n_steps:
                    break
                if i == self.save_per_steps:
                    self.init_c = self.c[i]
                    record.save_result(self.save_dir, self.c[i].cpu().numpy(), step)
                    # self.save_result(self.c[i], f"{cnt}_step")


    # def save_result(self, c: torch.Tensor, filename: str):
    #     c = c.cpu().numpy()
    #     np.save(
    #         self.npy_dir.joinpath(f"{filename}.npy"),
    #         (c - c.min()) / (c.max() - c.min())
    #     )


