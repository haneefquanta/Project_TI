# %%
import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
from Kitaev_realspace import Hamiltonian


mu = 2
N = 100
BC = 1
H = Hamiltonian(N, mu, 1, 1, BC)
H_m= build_HM_blocks(N, mu, 1, 1, BC, parity=1)
O_init = np.zeros(2*N, dtype=complex)
O_init[0] = 1.0

def lanczos(H, O_init):
    dim = H.shape[0]
    
    
    krylov_basis = []
    a_coeffs = []
    b_coeffs = [0.0]          # b_0 = 0 by convention
    
    # normalize seed
    O_n = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros(dim)    # O_{-1} = 0
    
    for n in range(dim):
        # apply L = i * H_M
        A = 2 * H @ O_n * 1j
        
        # diagonal coefficient
        a_n = np.real(O_n.conj() @ A)
        a_coeffs.append(a_n)
        
        # subtract projections
        A = A - a_n * O_n - b_coeffs[n] * O_prev 
        
        # off-diagonal coefficient
        b_next = np.linalg.norm(A)
        
        if b_next < 1e-7:    # ← convergence: basis exhausted
            print(f"Krylov space exhausted at K = {n+1}")
            break
        
        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n = A / b_next
        krylov_basis.append(O_n.copy())
    b_coeffs = b_coeffs[1:]
    return np.array(b_coeffs) #np.array(krylov_basis), np.array(a_coeffs), 

b_coeffs = lanczos(H_m    ,O_init)

betas = b_coeffs

n = np.arange(1, len(betas) + 1)

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(n[0::2], betas[0::2], 'o-', color='coral',   label='odd $b_n$',  markersize=5)
ax.plot(n[1::2], betas[1::2], 's-', color='steelblue', label='even $b_n$', markersize=5)

ax.set_xlabel('$n$', fontsize=13)
ax.set_ylabel('$b_n$', fontsize=13)
ax.set_title(
    rf'Lanczos coefficients — $\mu={mu}$, '
    rf'$N={N}$, BC: {BC}',
    fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
