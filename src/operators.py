import numpy as np
from scipy import sparse


def make_d_operator(N):
    diag = np.ones(N)
    diags = np.array([diag, -2 * diag, diag])
    d = sparse.spdiags(diags, [-1, 0, 1], N, N)
    return d


def make_D_operator(N, toarray=False):
    d = make_d_operator(N)
    D = sparse.kronsum(d, d)
    if toarray:
        return D.toarray()
    else:
        return D


def make_d4_operator(N):
    diag1 = -30 * np.ones(N)
    diag2 = 16 * np.ones(N)
    diag3 = -np.ones(N)
    diags = np.array([diag3, diag2, diag1, diag2, diag3])
    d4 = sparse.spdiags(diags, [-2, -1, 0, 1, 2], N, N)
    return d4


def make_D4_operator(N, toarray=False):
    d4 = (1 / 12) * make_d4_operator(N)
    D4 = sparse.kronsum(d4, d4)
    if toarray:
        return D4.toarray()
    else:
        return D4


def make_V_operator(X, Y, V_func, lam=0, toarray=False, dense=False):
    if dense:
        return V_func(X, Y, lam)
    else:
        V = sparse.diags(V_func(X, Y, lam).flatten(), (0))
        if toarray:
            return V.toarray()
        else:
            return V


def make_Dx_operator(N, toarray=False):
    d = make_d_operator(N)

    dx = sparse.kron(sparse.eye(N), d)

    if toarray:
        return dx.toarray()
    else:
        return dx


def make_Dy_operator(N, toarray=False):
    d = make_d_operator(N)

    dy = sparse.kron(d, sparse.eye(N))

    if toarray:
        return dy.toarray()
    else:
        return dy


def make_Dx4_operator(N, toarray=False):
    d4 = (1 / 12) * make_d4_operator(N)
    d4x = sparse.kron(sparse.eye(N), d4)

    if toarray:
        return d4x.toarray()
    else:
        return d4x


def make_Dy4_operator(N, toarray=False):
    d4 = (1 / 12) * make_d4_operator(N)
    d4y = sparse.kron(d4, sparse.eye(N))

    if toarray:
        return d4y.toarray()
    else:
        return d4y
