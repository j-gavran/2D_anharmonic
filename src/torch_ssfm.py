import torch
from torch.fft import fft2, fftshift, ifft2
from tqdm import tqdm

from problem_setup import normalize_psi


def solve_torch_split_step_fft(
    t, psi, V, P, ht, hr, method=1, im_time=False, renormalize_every=1, save_every=1, check_overflow=None
):
    save = []

    if im_time:
        time = 1
    else:
        time = 1j

    for i in tqdm(range(len(t))):
        if method == 1:
            psi_1 = torch.exp(-0.5 * time * V * ht) * psi
            psi_2 = torch.exp(-time * P * ht) * fftshift(fft2(psi_1))
            psi = torch.exp(-0.5 * time * V * ht) * ifft2(fftshift(psi_2))
        elif method == 2:
            psi_1 = torch.exp(-0.5 * time * P * ht) * fftshift(fft2(psi))
            psi_2 = torch.exp(-time * V * ht) * ifft2(fftshift(psi_1))
            psi_3 = torch.exp(-0.5 * time * P * ht) * fftshift(fft2(psi_2))
            psi = ifft2(fftshift(psi_3))
        else:
            raise ValueError

        if renormalize_every and i % renormalize_every == 0:
            psi = normalize_psi(psi, hr, hr, backend="torch")

        if save_every and i % save_every == 0:
            save.append(psi.cpu().numpy())

    return save


if __name__ == "__main__":
    from animations import animate
    from problem_setup import Discretization, HarmonicPotential
    from utils import probability_density

    grid = Discretization(N=512, L=15, Nt=20000, T=20, backend="torch")
    grid.init_all(n=0, a_x=5, a_y=0, flatten_psi=False)

    pot = HarmonicPotential(grid.X, grid.Y, lam=0.5)

    P = 0.5 * (grid.k_X ** 2 + grid.k_Y ** 2)

    print(grid.Nt // 500)

    saved_psi = solve_torch_split_step_fft(
        t=grid.t,
        psi=grid.psi,
        V=pot.V_dense(),
        P=P,
        ht=grid.ht,
        hr=grid.hr,
        method=1,
        im_time=False,
        renormalize_every=1,
        save_every=grid.Nt // 500,
    )

    animate(grid.X_np, grid.Y_np, probability_density(saved_psi, grid.N), speedup=1, save="ani")
