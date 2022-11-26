import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv

from problem_setup import normalize_psi


def crank_nicolson(T, V, hr, ht, n, which=0):
    """Crank-Nicolson formula with sparse matrix inverse. Used in split_step_cn.

    Parameters
    ----------
    T : sparse.csr_matrix
        Same as D. Use :math:`\frac{T}{2 \Delta x^2}` to get actual T.
    V : sparse.csr_matrix
    hr : float
    ht : float
    n : int
        Dimesnion of psi.
    which : int, optional
        Which exponent to use in split_step, by default 0.

    Returns
    -------
    sparse.csc.matrix
        Result of :math:`\frac{2 - iH \Delta t}{2 + iH \Delta t} ` where :math:`H = T + V`.

    Raises
    ------
    ValueError
        Which can be in {0, 1, 2} else raise ValueError.

    Note
    ----
    Inverse of matrix is O(n^3).

    """
    T = sparse.csc_matrix(T)
    V = sparse.csc_matrix(V)
    I = sparse.csc_matrix(sparse.eye(n))

    if which == 0:
        a = 2 * I - 1j * ht * (-0.5 * T / hr ** 2 + V)
        b = inv(2 * I + 1j * ht * (-0.5 * T / hr ** 2 + V))
    elif which == 1:
        a = 2 * I - 1j * ht * (-0.5 * T / hr ** 2)
        b = inv(2 * I + 1j * ht * (-0.5 * T / hr ** 2))
    elif which == 2:
        a = 2 * I - 1j * ht * V
        b = inv(2 * I + 1j * ht * V)
    else:
        raise ValueError

    c = a @ b

    return sparse.csc_matrix(c)


def solve_split_step_cn(t, psi, D, V, hr, split_step=1, renormalize_every=1, save_every=1):
    """Crank-Nicolson scheme using split step.

    Parameters
    ----------
    t : np.array
        Time interval with step ht.
    psi : np.array
        Starting psi with N^2 dim.
    D : sparse.csr_matrix
        Matrix D operator of shape N x N.
    V : sparse.csr_matrix
        Matrix V operator of shape N x N
    hr : float
        Position step.
    split_step : int, optional
        If default use fisrt order e^A e^B split step, else use second order e^A/2 e^B e^A/2, by default 1.
    renormalize : int, optional
        Renormalizes psi every int time steps, by default 1.
    save_every : int, optional
        Append new psi to list every int time steps, by default 1.

    Returns
    -------
    list
        Saved list of wave functions arrays.

    Raises
    ------
    ValueError
        If split_step not in {1, 2}.
    """

    dt = t[1] - t[0]
    save = []
    n = len(psi)

    if split_step == 1:
        cn = crank_nicolson(D, V, hr, dt, n, which=0)
    elif split_step == 2:
        cn1 = crank_nicolson(0.5 * D, 0, hr, dt, n, which=1)
        cn2 = crank_nicolson(0, V, hr, dt, n, which=2)
        cn3 = crank_nicolson(0.5 * D, 0, hr, dt, n, which=1)
        cn = cn1 @ cn2 @ cn3
    else:
        raise ValueError

    for i in range(len(t)):
        if split_step == 1:
            psi = cn @ psi

        if split_step == 2:
            psi = cn @ psi

        if renormalize_every and i % renormalize_every == 0:
            psi = normalize_psi(psi, hr, hr, backend="numba")

        if i % save_every == 0:
            print(i, np.sum(np.abs(psi) ** 2) * hr * hr)
            save.append(psi.copy())

    return save


def solve_euler(t, psi, T, V, hr, renormalize_every=1, save_every=1):
    dt = t[1] - t[0]
    save = []
    n = len(psi)

    I = sparse.eye(n)

    for i in range(len(t)):
        psi = (I - 1j * dt * (-0.5 * T / hr ** 2 + V)) @ psi

        if renormalize_every and i % renormalize_every == 0:
            psi = normalize_psi(psi, hr, hr, backend="numba")

        if i % save_every == 0:
            print(i, np.sum(np.abs(psi) ** 2) * hr * hr)
            save.append(psi)

        if np.sum(np.abs(psi)) > 1e6:
            raise OverflowError

    return save


if __name__ == "__main__":
    from animations import animate
    from operators import make_D4_operator
    from problem_setup import Discretization, HarmonicPotential
    from utils import plot_max_values, plot_norms, probability_density

    grid = Discretization(N=32, L=5, Nt=3000, T=3)
    grid.init_all(n=0, a_x=1, a_y=1, lam=0, flatten_psi=True)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0)

    D4 = make_D4_operator(grid.N, toarray=False)

    saved_psi = solve_split_step_cn(
        t=grid.t,
        psi=grid.psi,
        D=D4,
        V=pot.V_diag(),
        hr=grid.hr,
        renormalize_every=None,
        split_step=2,
        save_every=100,
    )

    plot_norms(saved_psi, grid.hr)
    plot_max_values(saved_psi, grid.N)

    animate(grid.X, grid.Y, probability_density(saved_psi, grid.N), speedup=1)
