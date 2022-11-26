import numpy as np
from tqdm import tqdm

from operators import make_D4_operator, make_D_operator
from utils import psi_norm


def solve_matrix_diff(t, psi, V, ht, hr, use_4th_order=True, check_overflow=2, save_every=1):
    save = []
    psi0, psi1 = psi, psi

    N = len(psi)
    n = int(np.sqrt(N))

    if use_4th_order:
        D = make_D4_operator(n)
    else:
        D = make_D_operator(n)

    for i in tqdm(range(len(t))):
        psi = psi0 + 1j * (ht / hr ** 2) * D @ psi1 - 2 * 1j * ht * V @ psi1

        psi0, psi1 = psi1, psi

        if check_overflow:
            if psi_norm(psi, hr) > check_overflow:
                print("Overflowed!")
                return None

        if save_every and i % save_every == 0:
            save.append(psi)

    return save


if __name__ == "__main__":
    from animations import animate
    from problem_setup import Discretization, HarmonicPotential
    from utils import plot_max_values, plot_norms, probability_density

    grid = Discretization(N=128, L=5, Nt=10000, T=10)
    grid.init_all(n=0, a_x=1, a_y=1, lam=0, flatten_psi=True)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    D4 = make_D4_operator(grid.N, toarray=False)

    saved_psi = solve_matrix_diff(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_diag(),
        ht=grid.ht,
        hr=grid.hr,
        check_overflow=True,
        save_every=100,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)
