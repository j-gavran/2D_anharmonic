# 2D anharmonski oscilator
## Zaključna naloga: višje računske metode
Final project for the course [Advanced computational methods](https://predmeti.fmf.uni-lj.si/VRM) 20/21.
### Solvers:
1. Euler, Crank-Nicolson - ```cn.py```
2. Finite difference methods - ```iter_diff.py``` & ```matrix_diff.py```
3. Alternating-direction implicit method - ```adi_sparse.py``` & ```adi_tridiag.py```
4. Split-step Fourier method - ```ssfm.py``` & ```torch_ssfm.py```  

```Python
from methods import Method # wrapper za razlicne solverje

L, T, N, Nt = 5, 10, 128, 10000 # parametri mreze
n, a_x, a_y, lam = 0, 1, 0, 0   # parametri potenciala

M = Method(N=N, Nt=Nt, method_name="tridiag_adi")
M.init_grid(L=L, T=T)
M.init_pot(n=n, a_x=a_x, a_y=a_y, lam=lam)

res = M.start_method(check_overflow=False, save_every=Nt // 100)
```
