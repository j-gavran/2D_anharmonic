import numpy as np
from scipy.fft import fftn, fftshift, ifftn
from tqdm import tqdm

from problem_setup import normalize_psi
from utils import psi_norm


def solve_split_step_fft(t, psi, V, P, ht, hr, method=1, im_time=False, check_overflow=2, save_every=1):
    save = []

    if im_time:
        time = 1
    else:
        time = 1j

    for i in tqdm(range(len(t))):
        if method == 1:
            psi_1 = np.exp(-0.5 * time * V * ht) * psi
            psi_2 = np.exp(-time * P * ht) * fftshift(fftn(psi_1, workers=-1))
            psi = np.exp(-0.5 * time * V * ht) * ifftn(fftshift(psi_2), workers=-1)
        elif method == 2:
            psi_1 = np.exp(-0.5 * time * P * ht) * fftshift(fftn(psi, workers=-1))
            psi_2 = np.exp(-time * V * ht) * ifftn(fftshift(psi_1), workers=-1)
            psi_3 = np.exp(-0.5 * time * P * ht) * fftshift(fftn(psi_2, workers=-1))
            psi = ifftn(fftshift(psi_3), workers=-1)
        else:
            raise ValueError

        if check_overflow:
            if psi_norm(psi, hr) > check_overflow:
                print("Overflowed!")
                return None

        if save_every and i % save_every == 0:
            save.append(psi)

        if im_time:
            psi = normalize_psi(psi, hr, hr)

    return save


if __name__ == "__main__":
    from animations import animate
    from problem_setup import Discretization, HarmonicPotential
    from utils import (
        calculate_E,
        plot_Es,
        plot_max_values,
        plot_norms,
        probability_density,
    )

    grid = Discretization(N=128, L=5, Nt=10, T=10)
    grid.init_all(n=0, a_x=1, a_y=0, lam=0, flatten_psi=False)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    P = 0.5 * (grid.k_X ** 2 + grid.k_Y ** 2)

    saved_psi = solve_split_step_fft(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_dense(),
        P=P,
        ht=grid.ht,
        hr=grid.hr,
        method=1,
        im_time=False,
        save_every=1,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)

    E, Ek, Ep = calculate_E(saved_psi, grid.hr, V=pot.V)

    plot_Es(E, Ek, Ep, norm=False)
