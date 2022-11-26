from numba import njit

from utils import njit_psi_norm


@njit
def solve_iter_diff(t, psi, V, ht, hr, check_overflow=2, save_every=1):
    """Solve with n + 1 and n - 1 split step as :math:`\psi_{ij}^{n+1} = \psi_{ij}^{n-1} - 2i H \Delta t \psi_{ij}^n`
    using second order finite differences for d^2/dx^2 and d^2/dy^2.

    Second order stencil:

                 u=ij+1
                   |
                   |
                   |
      l=i-1j------m=ij------r=i+1j
                   |
                   |
                   |
                 d=ij-1

    Parameters
    ----------
    t : np.array
        Time interval with step ht.
    psi : np.array
        Starting N x N psi.
    V : np.array
        2d potential.
    ht : float
        Time step.
    hr : float
        position step
    save_every : int, optional
        Append new psi to list every int time steps, by default 1.

    Returns
    -------
    list
        Saved list of wave functions arrays.
    """
    save = [psi.copy()]
    N = len(psi)

    psi0, psi1 = psi.copy(), psi.copy()

    for n in range(len(t)):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                m0, m1 = psi0[i, j], psi1[i, j]
                l, r = psi1[i - 1, j], psi1[i + 1, j]
                d, u = psi1[i, j - 1], psi1[i, j + 1]
                v = V[i, j]

                psi[i, j] = m0 - 2j * ht * ((-0.5 / hr ** 2) * (l + r + u + d - 4 * m1) + v * m1)

        psi0, psi1 = psi1.copy(), psi.copy()

        if check_overflow is not None:
            if njit_psi_norm(psi, hr) > check_overflow:
                print("Overflowed!")
                return None

        if save_every is not None and n % save_every == 0:
            save.append(psi.copy())

    return save[1:]


if __name__ == "__main__":
    from animations import animate
    from problem_setup import Discretization, HarmonicPotential
    from utils import plot_max_values, plot_norms, probability_density

    grid = Discretization(N=128, L=5, Nt=6601, T=10)  # <- 6601 je zadnji stabilni korak
    grid.init_all(n=0, a_x=1, a_y=1, lam=0, flatten_psi=False, zero_pad_psi=True)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    saved_psi = solve_iter_diff(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_dense(),
        ht=grid.ht,
        hr=grid.hr,
        save_every=100,
        check_overflow=None,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    # animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)
