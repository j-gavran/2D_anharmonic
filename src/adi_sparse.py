import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from tqdm import tqdm

from operators import (
    make_Dx4_operator,
    make_Dx_operator,
    make_Dy4_operator,
    make_Dy_operator,
)
from problem_setup import normalize_psi
from utils import psi_norm


def solve_sparse_adi(t, psi, V, ht, hr, use_4th_order=True, renormalize_every=None, check_overflow=1.1, save_every=1):
    N = len(psi)
    n = int(np.sqrt(N))
    save = []

    if use_4th_order:
        Dx, Dy = make_Dx4_operator(n), make_Dy4_operator(n)
    else:
        Dx, Dy = make_Dx_operator(n), make_Dy_operator(n)

    I = sparse.eye(N)

    M12 = (2j / ht) * I + (0.5 / hr ** 2) * Dx
    M1 = (2j / ht) * I + (0.5 / hr ** 2) * Dy

    A12 = (2j / ht) * I - (0.5 / hr ** 2) * Dy + V
    A1 = (2j / ht) * I - (0.5 / hr ** 2) * Dx + V

    solve12 = factorized(sparse.csc_matrix(M12))
    solve1 = factorized(sparse.csc_matrix(M1))

    for i in tqdm(range(len(t))):
        rhs12 = A12 @ psi
        psi12 = solve12(rhs12)

        rhs1 = A1 @ psi12
        psi = solve1(rhs1)

        if check_overflow:
            if psi_norm(psi, hr) > check_overflow:
                print("Overflowed!")
                return None

        if renormalize_every and i % renormalize_every == 0:
            psi = normalize_psi(psi, hr, hr)

        if save_every and i % save_every == 0:
            save.append(psi.copy())

    return save


if __name__ == "__main__":
    from problem_setup import Discretization, HarmonicPotential
    from utils import calculate_E, plot_Es, plot_max_values, plot_norms

    grid = Discretization(N=128, L=5, Nt=20000, T=10)
    grid.init_all(n=0, a_x=1, a_y=0, lam=0, flatten_psi=True)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    saved_psi = solve_sparse_adi(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_diag(),
        ht=grid.ht,
        hr=grid.hr,
        use_4th_order=False,
        renormalize_every=1,
        save_every=100,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    # animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)

    E, Ek, Ep = calculate_E(saved_psi, grid.hr, V=pot.V_dense())

    plot_Es(E, Ek, Ep, norm=False)
