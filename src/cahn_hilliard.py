import math
from pathlib import Path
import shutil
from typing import Optional, Literal

import numpy as np
from tqdm import tqdm
import scipy
import torch

from . import energy, record


class CahnHilliard:
    def __init__(
        self,
        c0: float,
        M: float,
        kappa: float,
        W: float,
        Nxy: int,
        Nz: Optional[int],
        dt: float = 0.1,
        dx: float = 1.0,
        noise: float = 0.1,
        n_steps: int = 1000,
        save_per_steps: int = 100,
        gif_interval: int = 10,
        output_dir: str = "output",
        seed: int = 42,
        device: Literal["numpy", "torch", "cuda", "cpu"] = "cuda",
    ):
        """Cahn-Hilliard方程式を解き、濃度分布の時間発展を計算する。

        Parameters
        ----------
        c0 : float
            初期濃度分布
        M : float
            移動度
        kappa : float
            _description_
        W : float
            _description_
        Nxy : int
            X軸、およびY軸の大きさ
        Nz : Optional[int]
            Z軸の大きさ
        dt : float, optional
            時間発展のΔt, by default 0.1
        dx : float, optional
            _description_, by default 1.0
        noise : float, optional
            初期構造に与えるノイズの大きさ, by default 0.1
        n_steps : int, optional
            計算step数, by default 1000
        save_per_steps : int, optional
            途中結果を保存するstep間隔, by default 100
        gif_interval : int, optional
            動画（GIFファイル）を作るときの1frame当たりのstep間隔, by default 10
        output_dir : str, optional
            結果を保存するディレクトリのpath, by default "output"
        seed : int, optional
            ランダムシード値, by default 42
        device : Literal[numpy, torch, cuda, cpu], optional
            計算デバイスの種類, by default "cuda"

        Raises
        ------
        ValueError
            _description_
        """
        self.c0 = c0
        self.M = M
        self.kappa = kappa
        self.W = W
        self.dt = dt
        self.Nxy = Nxy
        self.Nz = Nz
        self.dx = dx
        self.noise = noise
        self.n_steps = n_steps
        self.save_per_steps = save_per_steps
        self.gif_interval = gif_interval
        self.cycle = math.ceil(n_steps / save_per_steps)
        if (self.Nz == 1) or (self.Nz is None):
            self.dim = 2
        elif self.Nz > 1:
            self.dim = 3
        else:
            raise ValueError("Nz must be greater than 0")
        if device == "torch":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                print("GPUが認識できませんでした。CPU版に変更します。")
                self.device = "cpu"
        else:
            self.device = "numpy"

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.filename = f"c0_{self.c0:.3f}_M_{M}_kappa_{kappa}_W_{W}_dim_{self.dim}"
        self.save_dir = self.output_dir.joinpath(self.filename)
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir()
        self.save_dir.joinpath("npy").mkdir()
        self.save_dir.joinpath("img").mkdir()
        self.save_dir.joinpath("tif").mkdir()
        self.save_dir.joinpath("tmp").mkdir()

    def compute_c(self):
        self.initialize()
        step = 0
        for cycle in tqdm(range(self.cycle)):
            self.initialize_cycle()
            for i in range(1, self.save_per_steps + 1):
                self.time_evolution(i)
                step += 1
                if step > self.n_steps:
                    break
                if step % self.gif_interval == 0:
                    if self.device == "numpy":
                        record.save_tmp_fig(
                            self.c[i],
                            self.save_dir,
                            f"{self.filename}_{step}_step",
                            self.dim
                        )
                    else:
                        record.save_tmp_fig(
                            self.c[i].cpu().numpy(),
                            self.save_dir,
                            f"{self.filename}_{step}_step",
                            self.dim
                        )
                if i == self.save_per_steps:
                    self.init_c = self.c[i]
                    if self.device == "numpy":
                        record.save_result(
                            f"{self.filename}_{step}_step",
                            self.save_dir,
                            self.c[i]
                        )
                    else:
                        record.save_result(
                            f"{self.filename}_{step}_step",
                            self.save_dir,
                            self.c[i].cpu().numpy()
                        )
            del self.c
            if self.device == "cuda":
                torch.cuda.empty_cache()
        record.make_animation(self.save_dir, self.save_dir.joinpath("tmp"))

    def initialize(self):
        if self.device == "numpy":
            self.initialize_numpy()
            record.save_result(
                f"{self.filename}_0_step", self.save_dir, self.init_c,
            )
        else:
            self.initialize_torch()
            record.save_result(
                f"{self.filename}_0_step",
                self.save_dir, self.init_c.cpu().numpy(),
            )
        # self.L = self.N * self.dx

    def initialize_torch(self):
        if self.dim == 2:
            self.init_c = self.c0 + torch.tensor(
                self.noise * np.random.standard_normal((self.Nxy, self.Nxy)),
                device=self.device
            )
            self.c_hat = torch.zeros(
                (self.Nxy, self.Nxy),
                dtype=torch.complex64,
                device=self.device
            )
            self.dfdc_hat = torch.zeros(
                (self.Nxy, self.Nxy),
                dtype=torch.complex64,
                device=self.device
            )
            kx = ky = torch.fft.fftfreq(self.Nxy, d=self.dx) * 2 * np.pi
            kx, ky = torch.meshgrid(kx, ky, indexing="ij")
            self.K = torch.stack((kx, ky)).to(self.device)
            self.c0 = self.init_c.cpu().numpy().sum() / (self.Nxy**2)
        elif self.dim == 3:
            self.init_c = self.c0 + torch.tensor(
                self.noise * np.random.standard_normal(
                    (self.Nxy, self.Nxy, self.Nz)
                ),
                device=self.device
            )
            self.c_hat = torch.zeros(
                (self.Nxy, self.Nxy, self.Nz),
                dtype=torch.complex64,
                device=self.device
            )
            self.dfdc_hat = torch.zeros(
                (self.Nxy, self.Nxy, self.Nz),
                dtype=torch.complex64,
                device=self.device
            )
            kx = ky = torch.fft.fftfreq(self.Nxy, d=self.dx) * 2 * np.pi
            kz = torch.fft.fftfreq(self.Nz, d=self.dx) * 2 * np.pi
            kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")
            K = torch.stack((kx, ky, kz)).to(self.device)
            self.c0 = self.init_c.cpu().numpy().sum() / (self.Nxy**2 * self.Nz)
        else:
            raise ValueError("dim must be 2 or 3")
        self.k_max_dealias = kx.max() * 2.0 / 3.0
        self.dealias = (torch.abs(K[0]) < self.k_max_dealias) * (torch.abs(K[1]) < self.k_max_dealias)
        record.save_result(
            f"{self.filename}_0_step",
            self.save_dir,
            self.init_c.cpu().numpy(),
        )
        self.K2 = torch.sum(K * K, axis=0, dtype=torch.float16)
        self.c_hat[:] = torch.fft.fftn(self.init_c)

    def initialize_numpy(self):
        if self.dim == 2:
            self.init_c = self.c0 + self.noise * np.random.standard_normal((self.Nxy, self.Nxy))
            self.c_hat = np.zeros(
                (self.Nxy, self.Nxy),
                dtype=np.complex64
            )
            self.dfdc_hat = np.zeros(
                (self.Nxy, self.Nxy),
                dtype=np.complex64
            )
            kx = ky = np.fft.fftfreq(self.Nxy, d=self.dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, indexing="ij"),
                dtype=np.float16
            )
            self.c0 = self.init_c.sum() / (self.Nxy**2)
        elif self.dim == 3:
            self.init_c = self.c0 + self.noise * np.random.standard_normal((self.Nxy, self.Nxy, self.Nz))
            self.c_hat = np.zeros(
                (self.Nxy, self.Nxy, self.Nz),
                dtype=np.complex64
            )
            self.dfdc_hat = np.zeros(
                (self.Nxy, self.Nxy, self.Nz),
                dtype=np.complex64
            )
            kx = ky = np.fft.fftfreq(self.Nxy, d=self.dx) * 2 * np.pi
            kz = np.fft.fftfreq(self.Nz, d=self.dx) * 2 * np.pi
            self.K = np.array(
                np.meshgrid(kx, ky, kz, indexing="ij"),
                dtype=np.float16
            )
            self.c0 = self.init_c.sum() / (self.Nxy**2 * self.Nz)
        else:
            raise ValueError("dim must be 2 or 3")
        self.k_max_dealias = kx.max() * 2.0 / 3.0
        self.dealias = np.array(
            (np.abs(self.K[0]) < self.k_max_dealias)
            * (np.abs(self.K[1]) < self.k_max_dealias),
            dtype=bool
        )
        self.init_c = self.init_c.astype(np.float16)
        self.K2 = np.sum(self.K * self.K, axis=0, dtype=np.float16)
        self.c_hat[:] = scipy.fftpack.fftn(self.init_c)

    def initialize_cycle(self):
        if self.device == "numpy":
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
        else:
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

    def time_evolution(self, idx):
        if self.device == "numpy":
            self.dfdc_hat[:] = scipy.fftpack.fftn(
                energy.dfdc(self.c[idx - 1], self.W)
            )
            self.dfdc_hat *= self.dealias
            self.c_hat[:] = (
                (self.c_hat - self.dt * self.K2 * self.M * self.dfdc_hat)
                / (1 + self.dt * self.M * self.kappa * self.K2**2)
            )
            self.c[idx] = scipy.fftpack.ifftn(self.c_hat).real
        else:
            self.dfdc_hat[:] = torch.fft.fftn(
                energy.dfdc(self.c[idx - 1], self.W)
            )
            self.dfdc_hat *= self.dealias
            self.c_hat[:] = (
                (self.c_hat - self.dt * self.K2 * self.M * self.dfdc_hat)
                / (1 + self.dt * self.M * self.kappa * self.K2**2)
            )
            self.c[idx] = torch.fft.ifftn(self.c_hat).real
