import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def psi_norms(psi_lst, hr):
    return np.array([np.sum(np.abs(psi) ** 2) * hr * hr for psi in psi_lst])


def psi_norm(psi, hr, normalize_to_psi=False):
    if normalize_to_psi:
        if len(psi.shape) == 1:
            N = len(psi)
        else:
            N = psi.shape[0] * psi.shape[1]
    else:
        N = 1

    return np.sum(np.abs(psi) ** 2) * hr * hr / N


@njit
def njit_psi_norm(psi, hr, normalize_to_psi=False):
    if normalize_to_psi:
        if len(psi.shape) == 1:
            N = len(psi)
        else:
            N = psi.shape[0] * psi.shape[1]
    else:
        N = 1

    return np.sum(np.abs(psi) ** 2) * hr * hr / N


def probability_density(psi_lst, N):
    return np.array([np.abs(i.reshape(N, N)) ** 2 for i in psi_lst])


def max_psi_values(psi_lst, N):
    prob = probability_density(psi_lst, N)
    return np.array([np.max(psi) for psi in prob])


def plot_max_values(psi_lst, N):
    max_vals = max_psi_values(psi_lst, N)
    plt.plot(range(len(max_vals)), max_vals)
    plt.show()


def plot_norms(psi_lst, hr, minus_one=True):
    norms = psi_norms(psi_lst, hr)
    plt.plot(range(len(norms)), (1 - norms) if minus_one else norms)
    plt.show()


def calculate_E(psi_lst, hr, V, D=None):

    E_lst, E_k_lst, E_p_lst = [], [], []

    if D is None:
        for psi in psi_lst:
            if len(psi.shape) != 2:
                psi = psi.reshape(V.shape)
            E_k_lst.append(np.sum(0.5 * np.abs(np.gradient(psi, hr, hr, edge_order=2)) ** 2) * hr ** 2)
            E_p_lst.append(np.sum(V * np.abs(psi) ** 2) * hr ** 2)
            E_lst.append(E_k_lst[-1] + E_p_lst[-1])
    else:
        T = (-0.5 / hr ** 2) * D
        for psi in psi_lst:
            E_p = np.conj(psi) @ V @ psi
            E_k = np.conj(psi) @ T @ psi

            E_k_lst.append(np.real(np.sum(E_k)) * hr ** 2)
            E_p_lst.append(np.real(np.sum(E_p)) * hr ** 2)
            E_lst.append(E_k_lst[-1] + E_p_lst[-1])

    return E_lst, E_k_lst, E_p_lst


def calculate_xy(psi_lst, X, Y, hr):
    x_lst, y_lst = [], []
    for psi in psi_lst:
        if len(psi.shape) != 2:
            psi = psi.reshape(X.shape)

        x_lst.append(np.sum(X * np.abs(psi) ** 2) * hr ** 2)
        y_lst.append(np.sum(Y * np.abs(psi) ** 2) * hr ** 2)

    return x_lst, y_lst


def plot_Es(E, E_k, E_p, norm=False, mode=1, axs=None, show=True, lw=1):
    x = range(len(E))

    if mode == 1:
        if axs is None:
            fig, axs = plt.subplots(1, 4)
        axs[1].plot(x, E_k)
        axs[2].plot(x, E_p)
        axs[1].set_title("$E_k$")
        axs[2].set_title("$E_p$")
        axs[3].plot(x, E_p / np.max(np.abs(E_p)) if norm else E_p, label="$V$", lw=lw)
        axs[3].plot(x, E_k / np.max(np.abs(E_k)) if norm else E_k, label="$T$", lw=lw)
        axs[3].legend()
    elif mode == 2:
        if axs is None:
            fig, axs = plt.subplots(1, 2)
        axs[1].plot(x, E_p / np.max(np.abs(E_p)) if norm else E_p, label="$V$", lw=lw)
        axs[1].plot(x, E_k / np.max(np.abs(E_k)) if norm else E_k, label="$T$", lw=lw)
        axs[1].legend()
    elif mode == 3:
        if axs is None:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]

    axs[0].plot(x, E, c="C3", lw=lw)

    if show:
        plt.tight_layout()
        plt.show()
    else:
        return axs
