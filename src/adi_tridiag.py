import numpy as np
from scipy.linalg import solve_banded
from tqdm import tqdm

from problem_setup import normalize_psi


def solve_adi_tridiag(t, psi, V, ht, hr, renormalize_every=None, check_overflow=2, save_every=1):
    N = len(psi)
    save = [psi.copy()]
    a, b = 1j / (ht / 2), -1 / (2 * hr ** 2)
    a_, b_ = a * np.ones(N), b * np.ones(N)

    A = np.zeros((3, N), dtype=np.complex128)
    A[0, 1:] = -b_[:-1]
    A[1, :] = a_ + 2 * b_
    A[2, :-1] = -b_[1:]

    psi12 = np.zeros(psi.shape, dtype=np.complex128)

    for n in tqdm(range(len(t))):

        for i in range(1, N - 1):
            rhs = b * (psi[i + 1, :] - 2 * psi[i, :] + psi[i - 1, :]) + a * psi[i, :] + V[i, :] * psi[i, :]
            psi12[i, :] = solve_banded((1, 1), A, rhs, check_finite=False)

        for j in range(1, N - 1):
            rhs = b * (psi12[:, j + 1] - 2 * psi12[:, j] + psi12[:, j - 1]) + a * psi12[:, j] + V[:, j] * psi12[:, j]
            psi[:, j] = solve_banded((1, 1), A, rhs, check_finite=False)

        if check_overflow:
            if np.abs(np.max(save[0]) - np.max(psi)) > check_overflow:
                print("Overflowed!")
                return None

        if renormalize_every and i % renormalize_every == 0:
            psi = normalize_psi(psi, hr, hr)

        if save_every and n % save_every == 0:
            save.append(psi.copy())

    return save


if __name__ == "__main__":
    from animations import animate
    from problem_setup import Discretization, HarmonicPotential
    from utils import plot_max_values, plot_norms, probability_density

    grid = Discretization(N=128, L=5, Nt=11250, T=10)
    grid.init_all(n=0, a_x=1, a_y=0, lam=0, flatten_psi=False, zero_pad_psi=True)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    saved_psi = solve_adi_tridiag(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_dense(),
        ht=grid.ht,
        hr=grid.hr,
        save_every=500,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)
