import math

import numpy as np
import torch
from numba import njit
from scipy import sparse
from scipy.special import hermite


class HarmonicPotential:
    def __init__(self, X, Y, lam=0):
        self.X, self.Y = X, Y
        self.lam = lam
        self.V = None

    def V_dense(self):
        self.V = 0.5 * self.X ** 2 + 0.5 * self.Y ** 2 + self.lam * self.X ** 2 * self.Y ** 2
        return self.V

    def V_diag(self, toarray=False):
        self.V = sparse.diags(self.V_dense().flatten(), (0))
        if toarray:
            return self.V.toarray()
        else:
            return self.V

    def __call__(self, *args, **kwargs):
        return self.V_func(*args, **kwargs)


class Discretization:
    def __init__(self, N, L, Nt=None, T=None, backend="np"):
        self.N, self.L = N, L
        self.hr = 2 * L / (N - 1)

        if Nt and T:
            self.Nt, self.t_end = Nt, T
            self.t = np.linspace(0, T, Nt)
            self.ht = self.t[1] - self.t[0]
        else:
            self.Nt, self.T, self.t, self.ht = None, None, None, None

        self.backend = backend

        if backend == "torch":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif backend == "np":
            self.device = None
        else:
            raise ValueError

        self.X, self.Y = None, None
        self.k_X, self.k_Y = None, None
        self.psi = None

    def init(self, n, a_x, a_y, lam):
        self.a_x, self.a_y = a_x, a_y
        self.hermite_N = n
        self.lam = lam

    def init_all(self, n=0, a_x=0, a_y=0, lam=0, **psi_kwargs):
        self.init(n, a_x, a_y, lam)
        self.xy_grid()
        self.kxky_grid()
        self.starting_2d_psi(**psi_kwargs)

    def xy_grid(self):
        a = np.linspace(-self.L, self.L, self.N)
        self.X, self.Y = np.meshgrid(a, a)
        self.X_np, self.Y_np = self.X, self.Y

        if self.backend == "torch":
            a = torch.linspace(-self.L, self.L, self.N, device=self.device)
            self.X, self.Y = torch.meshgrid(a, a)

        return self.X, self.Y

    def kxky_grid(self, m=1):
        k_r = np.arange(0, self.N / 2, 1)
        k_l = np.arange(-self.N / 2, 0, 1)
        k = np.concatenate((k_l, k_r))

        if self.backend == "np":
            self.k_X, self.k_Y = np.meshgrid(k * m * np.pi / self.L, k * m * np.pi / self.L)
        elif self.backend == "torch":
            k = torch.from_numpy(k.astype(np.float32)).to(self.device)
            self.k_X, self.k_Y = torch.meshgrid(k * m * np.pi / self.L, k * m * np.pi / self.L)

        return self.k_X, self.k_Y

    def _starting_1d_psi(self, X, a=0):
        H = hermite(self.hermite_N)
        return (
            1
            / (np.pi ** (1 / 4) * np.sqrt(math.factorial(self.hermite_N) * 2 ** self.hermite_N))
            * H(X - a)
            * np.exp(-0.5 * (X - a) ** 2)
        )

    def starting_2d_psi(self, flatten_psi=False, normalize_psi=False, zero_pad_psi=False, **normalize_kwargs):
        psi_x = self._starting_1d_psi(self.X_np, self.a_x)
        psi_y = self._starting_1d_psi(self.Y_np, self.a_y)

        if zero_pad_psi:
            psi_xy = np.zeros((self.N, self.N))
            psi_xy[1:-1, 1:-1] = (psi_x * psi_y)[1:-1, 1:-1]
        else:
            psi_xy = psi_x * psi_y

        if flatten_psi:
            psi_xy = psi_xy.flatten()

        if normalize_psi:
            dx, dy = self.X_np[1, 1] - self.X_np[0, 0], self.Y_np[1, 1] - self.Y_np[0, 0]
            psi_xy = normalize_psi(psi_xy, dx, dy, **normalize_kwargs)

        if self.backend == "np":
            self.psi = psi_xy.astype(np.complex128)
        elif self.backend == "torch":
            self.psi = torch.from_numpy(psi_xy.astype(np.complex64)).to(self.device)

        return self.psi


def _normalize_psi(psi, dx, dy, backend):
    if backend == "np":
        renorm_factor = np.sum(np.abs(psi) ** 2) * dx * dy
        psi = psi / np.sqrt(renorm_factor)
    elif backend == "torch":
        renorm_factor = torch.sum(torch.abs(psi) ** 2) * dx * dy
        psi = psi / torch.sqrt(renorm_factor)
    elif backend == "numba":
        psi = _normalize_psi_njit(psi, dx, dy)
    else:
        raise ValueError

    return psi


@njit
def _normalize_psi_njit(psi, dx, dy):
    renorm_factor = np.sum(np.abs(psi) ** 2) * dx * dy
    psi = psi / np.sqrt(renorm_factor)
    return psi


def normalize_psi(psi, dx, dy, backend="np"):
    return _normalize_psi(psi, dx, dy, backend=backend)


if __name__ == "__main__":

    grid = Discretization(N=10, L=1, backend="torch")

    print(grid.init_all(n=0, a_x=0, a_y=0, lam=0))
