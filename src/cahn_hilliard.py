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
        c0: float | np.ndarray | torch.Tensor,
        M: float,
        kappa: float,
        W: float,
        Nx: int,
        Ny: int,
        Nz: Optional[int],
        dt: float = 0.1,
        dx: float = 1.0,
        noise: float = 0.1,
        n_steps: int = 1000,
        save_per_steps: int = 100,
        gif_interval: int = 10,
        output_dir: str = "output",
        seed: Optional[int] = 42,
        device: Literal["numpy", "torch", "cuda", "cpu"] = "cuda",
    ):
        """Cahn-Hilliard方程式を解き、濃度分布の時間発展を計算する。

        Parameters
        ----------
        c0 : float | np.ndarray | torch.Tensor
            初期濃度分布。floatの場合は濃度の平均値、np.ndarray、torch.Tensorの場合は濃度分布の配列を指定する。
        M : float
            移動度
        kappa : float
            _description_
        W : float
            _description_
        Nx : int
            X軸の大きさ
        Ny : int
            Y軸の大きさ
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
        # self.c0 = c0
        # ランダムシードの固定
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # 変数の設定
        self.M = M
        self.kappa = kappa
        self.W = W
        self.dt = dt
        self.dx = dx
        self.noise = noise
        self.n_steps = n_steps
        self.save_per_steps = save_per_steps
        self.gif_interval = gif_interval
        self.cycle = math.ceil(n_steps / save_per_steps)
        # 計算サイズの設定
        # 濃度分布の設定
        if isinstance(c0, torch.Tensor):
            self.Ny, self.Nx = c0.shape[0], c0.shape[1]
            if len(c0.shape) == 3:
                self.Nz = c0.shape[2]
                self.dim = 3
            else:
                self.Nz = 1
                self.dim = 2
            self.c0 = c0.cpu().numpy().sum() / (self.Nx * self.Ny * self.Nz)
            self.init_c = c0
            self.c = torch.repeat_interleave(
                c0[torch.newaxis, ...], self.save_per_steps + 1, axis=0
            )
            self.c_shape = c0.shape
        elif isinstance(c0, np.ndarray):
            self.Ny, self.Nx = c0.shape[0], c0.shape[1]
            if len(c0.shape) == 3:
                self.Nz = c0.shape[2]
                self.dim = 3
            else:
                self.Nz = 1
                self.dim = 2
            self.c0 = c0.sum() / (self.Nx * self.Ny * self.Nz)
            self.init_c = c0
            self.c = np.repeat(
                c0[np.newaxis, ...], self.save_per_steps + 1, axis=0
            )
            self.c_shape = c0.shape
        else:
            self.Nx = int(Nx)
            self.Ny = int(Ny)
            if (Nz is None) or (Nz == 1):
                self.Nz = 1
                self.dim = 2
                self.c_shape = (self.Ny, self.Nx)
            elif Nz > 1:
                self.Nz = int(Nz)
                self.dim = 3
                self.c_shape = (self.Ny, self.Nx, self.Nz)
            else:
                raise ValueError("Nz must be greater than 0")
            self.c0 = c0
            self.init_c = (
                self.c0
                + self.noise
                * np.random.standard_normal(self.c_shape)
            )
            self.c = np.empty(
                (self.save_per_steps + 1, *self.c_shape),
                dtype=np.float16
            )
            self.c[0] = self.init_c
        # 計算デバイスの設定
        if device == "torch":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.init_c = torch.tensor(
                self.init_c, device=self.device, dtype=torch.float16
            )
            self.c = torch.tensor(
                self.c, device=self.device, dtype=torch.float16
            )
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                print("GPUが認識できませんでした。CPU版に変更します。")
                self.device = "cpu"
            self.init_c = torch.tensor(
                self.init_c, device=self.device, dtype=torch.float16
            )
            self.c = torch.tensor(
                self.c, device=self.device, dtype=torch.float16
            )
        else:
            self.device = "numpy"
            self.init_c = self.init_c.astype(np.float16)
            self.c = self.c.astype(np.float16)

        # 計算結果の保存先の設定
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

        if self.device == "numpy":
            record.save_result(
                f"{self.filename}_0_step",
                self.save_dir,
                self.init_c,
            )
        else:
            record.save_result(
                f"{self.filename}_0_step",
                self.save_dir,
                self.init_c.cpu().numpy(),
            )

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
            # メモリ解放
            del self.c
            if self.device == "cuda":
                torch.cuda.empty_cache()
        record.make_animation(self.save_dir, self.save_dir.joinpath("tmp"))

    def initialize(self):
        if self.device == "numpy":
            self.initialize_numpy()
        else:
            self.initialize_torch()

    def initialize_torch(self):
        self.c_hat = torch.zeros(
            self.c_shape,
            dtype=torch.complex64,
            device=self.device
        )
        self.dfdc_hat = torch.zeros(
            self.c_shape,
            dtype=torch.complex64,
            device=self.device
        )
        kx = torch.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        ky = torch.fft.fftfreq(self.Ny, d=self.dx) * 2 * np.pi
        kz = torch.fft.fftfreq(self.Nz, d=self.dx) * 2 * np.pi
        if self.dim == 2:
            ky, kx = torch.meshgrid(ky, kx, indexing="ij")
            K = torch.stack((ky, kx)).to(torch.float16).to(self.device)
            kx_max_dealias = kx.max() * 2.0 / 3.0
            ky_max_dealias = ky.max() * 2.0 / 3.0
            self.dealias = (
                (torch.abs(K[0]) < ky_max_dealias)
                * (torch.abs(K[1]) < kx_max_dealias)
            )
        elif self.dim == 3:
            ky, kx, kz = torch.meshgrid(ky, kx, kz, indexing="ij")
            K = torch.stack((ky, kx, kz)).to(torch.float16).to(self.device)
            kx_max_dealias = kx.max() * 2.0 / 3.0
            ky_max_dealias = ky.max() * 2.0 / 3.0
            kz_max_dealias = kz.max() * 2.0 / 3.0
            self.dealias = (
                (torch.abs(K[0]) < ky_max_dealias)
                * (torch.abs(K[1]) < kx_max_dealias)
                * (torch.abs(K[2]) < kz_max_dealias)
            )
        else:
            raise ValueError("dim must be 2 or 3")
        self.K2 = torch.sum(K * K, axis=0, dtype=torch.float16)
        self.c_hat[:] = torch.fft.fftn(
            self.init_c.to(torch.float32)
        )

    def initialize_numpy(self):
        self.c_hat = np.zeros(
            self.c_shape,
            dtype=np.complex64
        )
        self.dfdc_hat = np.zeros(
            self.c_shape,
            dtype=np.complex64
        )
        kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.Ny, d=self.dx) * 2 * np.pi
        kz = np.fft.fftfreq(self.Nz, d=self.dx) * 2 * np.pi
        if self.dim == 2:
            kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi
            ky = np.fft.fftfreq(self.Ny, d=self.dx) * 2 * np.pi
            K = np.array(
                np.meshgrid(ky, kx, indexing="ij"),
                dtype=np.float16
            )
            kx_max_dealias = kx.max() * 2.0 / 3.0
            ky_max_dealias = ky.max() * 2.0 / 3.0
            self.dealias = (
                (np.abs(K[0]) < ky_max_dealias)
                * (np.abs(K[1]) < kx_max_dealias)
            )
        elif self.dim == 3:
            kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi
            ky = np.fft.fftfreq(self.Ny, d=self.dx) * 2 * np.pi
            kz = np.fft.fftfreq(self.Nz, d=self.dx) * 2 * np.pi
            K = np.array(
                np.meshgrid(ky, kx, kz, indexing="ij"),
                dtype=np.float16
            )
            kx_max_dealias = kx.max() * 2.0 / 3.0
            ky_max_dealias = ky.max() * 2.0 / 3.0
            kz_max_dealias = kz.max() * 2.0 / 3.0
            self.dealias = (
                (np.abs(K[0]) < ky_max_dealias)
                * (np.abs(K[1]) < kx_max_dealias)
                * (np.abs(K[2]) < kz_max_dealias)
            )
        else:
            raise ValueError("dim must be 2 or 3")
        self.K2 = np.sum(K * K, axis=0, dtype=np.float16)
        self.c_hat[:] = scipy.fftpack.fftn(self.init_c)

    def initialize_cycle(self):
        if self.device == "numpy":
            if self.dim == 2:
                self.c = np.empty(
                    (self.save_per_steps + 1, self.Ny, self.Nx),
                    dtype=np.float16
                )
            if self.dim == 3:
                self.c = np.empty(
                    (self.save_per_steps + 1, self.Ny, self.Nx, self.Nz),
                    dtype=np.float16
                )
        else:
            if self.dim == 2:
                self.c = torch.empty(
                    (self.save_per_steps + 1, self.Ny, self.Nx),
                    dtype=torch.float32,
                    device=self.device
                )
            if self.dim == 3:
                self.c = torch.empty(
                    (self.save_per_steps + 1, self.Ny, self.Nx, self.Nz),
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
