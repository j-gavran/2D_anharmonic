import pickle

import numpy as np

from adi_sparse import solve_sparse_adi
from adi_tridiag import solve_adi_tridiag
from iter_diff import solve_iter_diff
from matrix_diff import solve_matrix_diff
from problem_setup import Discretization, HarmonicPotential
from ssfm import solve_split_step_fft
from torch_ssfm import solve_torch_split_step_fft
from utils import calculate_E, max_psi_values, psi_norms


class Method:
    def __init__(self, N, Nt, method_name):
        self.N = N
        self.Nt = Nt
        self.method_name = method_name

        if method_name in ["tridiag_adi", "iter_diff", "ssfm", "torch_ssfm"]:
            self.flatten = False
        else:
            self.flatten = True

        self.solvers = {
            "sparse_adi": solve_sparse_adi,
            "tridiag_adi": solve_adi_tridiag,
            "iter_diff": solve_iter_diff,
            "matrix_diff": solve_matrix_diff,
            "ssfm": solve_split_step_fft,
            "torch_ssfm": solve_torch_split_step_fft,
        }

        self.grid_kwargs, self.pot_kwargs = None, None
        self.res = None
        self.norm, self.max_val, self.E = None, None, None

    def init_grid(self, **grid_kwargs):
        self.grid_kwargs = grid_kwargs
        self.grid = Discretization(N=self.N, Nt=self.Nt, **grid_kwargs)

    def init_pot(self, **pot_kwargs):
        self.pot_kwargs = pot_kwargs

        if self.flatten:
            self.grid.init_all(**pot_kwargs, flatten_psi=True)
        else:
            self.grid.init_all(**pot_kwargs, flatten_psi=False)

        self.pot = HarmonicPotential(self.grid.X, self.grid.Y, lam=pot_kwargs["lam"])

        if self.flatten:
            self.pot.V_diag()
        else:
            self.pot.V_dense()

    def start_method(self, save_every=1, check_overflow=False, **kwargs):
        if self.method_name == "ssfm" or self.method_name == "torch_ssfm":
            kwargs.update({"P": 0.5 * (self.grid.k_X ** 2 + self.grid.k_Y ** 2)})

        res = self.solvers[self.method_name](
            t=self.grid.t,
            psi=self.grid.psi,
            V=self.pot.V,
            ht=self.grid.ht,
            hr=self.grid.hr,
            save_every=save_every,
            check_overflow=check_overflow,
            **kwargs,
        )

        self.res = res
        return res

    def calculate_util_values(self):
        norm = psi_norms(self.res, self.grid.hr)
        max_val = max_psi_values(self.res, self.grid.N)
        E = calculate_E(self.res, self.grid.hr, V=self.pot.V_dense())

        self.norm, self.max_val, self.E = norm, max_val, E

        return norm, max_val, E

    def find_stable_time_step(self, Nt_min, Nt_max, it=10, check_overflow=2, **kwargs):
        for i in range(it):
            nt = (Nt_max + Nt_min) // 2

            self.Nt = nt
            self.init_grid(**self.grid_kwargs)
            self.init_pot(**self.pot_kwargs)

            self.start_method(check_overflow=check_overflow, **kwargs)

            if self.res is None:
                Nt_min = nt
            else:
                stable = nt
                Nt_max = nt

            if np.abs(Nt_max - Nt_min) == 0:
                stable = nt
                break

        self.Nt = stable
        return stable

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


class SaverLoader:
    def __init__(self, method_obj=None, save_dir="saved_results/"):
        self.method = method_obj
        self.save_dir = save_dir

    def save(self):
        name = lambda q: f"{q}_{self.method.method_name}_{self.method.N}_{self.method.Nt}.p"

        if self.method.norm is not None:
            pickle.dump(self.method.norm, open(name("norm"), "rb"))
        if self.method.max_val is not None:
            pickle.dump(self.method.max_val, open(name("max"), "rb"))
        if self.method.E is not None:
            pickle.dump(self.method.E, open(name("E"), "rb"))

    def load(self, name):
        return pickle.load(open(self.save_dir + f"{name}.p", "rb"))


class MultiNSaverLoader:
    def __init__(self, method_obj_lst=None, save_dir="saved_results/"):
        self.method_lst = method_obj_lst
        self.save_dir = save_dir

    def save(self):
        method = self.method_lst[0]
        name = lambda q: f"{q}_{method.method_name}_multi_{method.Nt}.p"

        norms, max_vals, Es = [], [], []

        for m in self.method_lst:
            if m.norm is not None:
                norms.append(m.norm)
            if m.max_val is not None:
                max_vals.append(m.max_val)
            if m.E is not None:
                Es.append(m.E)

        if len(norms) != 0:
            pickle.dump(norms, open(name("norm"), "rb"))
        if len(max_vals) != 0:
            pickle.dump(norms, open(name("max"), "rb"))
        if len(Es) != 0:
            pickle.dump(norms, open(name("E"), "rb"))

    def load(self, name):
        return pickle.load(open(self.save_dir + f"{name}.p", "rb"))


if __name__ == "__main__":
    L, T, N = 5, 10, 128

    n, a_x, a_y, lam = 0, 1, 0, 0

    M = Method(N=N, Nt=N, method_name="tridiag_adi")
    M.init_grid(L=L, T=T)
    M.init_pot(n=n, a_x=a_x, a_y=a_y, lam=lam)

    M.find_stable_time_step(Nt_min=10000, Nt_max=10001, it=1, save_every=None)

    M.init_grid(L=L, T=T)
    M.init_pot(n=n, a_x=a_x, a_y=a_y, lam=lam)

    M.start_method(check_overflow=False, save_every=200)

    from utils import plot_max_values

    plot_max_values(M.res, M.grid.N)
