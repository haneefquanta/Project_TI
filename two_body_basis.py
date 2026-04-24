# %%
import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
from Kitaev_realspace import Hamiltonian
import matplotlib.cm as cm

mu_values =np.linspace(0.5,3.5,50)
mu_values = [1,2,3]
t= 1
delta = 1


N = 20
BC = 0
O_init = np.zeros((2*N,2*N), dtype=complex)
site = 0    # mi
w = 50
#w =50


#O_init[0,N] = 1
#O_init[N,0] = -1

O_init[0,2*N - 1] = -1
O_init[2*N -1 ,0] = 1







def lanczos(H, O_init,window=w):
    dim    = H.shape[0]
    krylov_basis = []
    a_coeffs = []
    b_coeffs = [0.0]

    

    O_n = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros((dim, dim))

    for n in range(window):
        
        

        A = (H @ O_n - O_n @ H) * 1j
        
        
        a_n = np.real(O_n.conj() @ A)         # ← fixed inner product
        a_coeffs.append(a_n)

        A = A - a_n * O_n - b_coeffs[n] * O_prev
        
        #   reorthogonalization 
        for O_k in krylov_basis[-window:]: 
            A = A - np.sum(O_k.conj() * A) * O_k
        for O_k in krylov_basis[-window:]:   
            A = A - np.sum(O_k.conj() * A) * O_k
        
        
        b_next = np.linalg.norm(A)

        if b_next < 1e-3:
            print(f"Krylov space exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

       

    b_coeffs = b_coeffs[1:]
    return np.array(b_coeffs) , krylov_basis


results = {}
results_kb = {}
for mu in mu_values:
    print(f"Running μ = {mu} ...", end="  ")
    H_m = build_HM_blocks(N, mu,t,delta , BC,-1)
    b ,K_b  = lanczos(H_m, O_init) 
    results[mu] = b 
    results_kb[mu] = K_b  
    print(results[mu])                      
    print(f"done  (K = {len(b)})")
   
colors = cm.plasma(np.linspace(0.1, 0.9, len(mu_values)))







fig, ax = plt.subplots(figsize=(10, 5))
for (mu, b), color in zip(results.items(), colors):
    label = rf'$\mu = {mu}$' + (' ← critical' if mu == 2.0 else '')
    n_b   = np.arange(1, len(b) + 1)
    ax.scatter(n_b, b, color=color, s=4, alpha=0.6, label=label, zorder=2)
    ax.plot(n_b, b, '-', color=color, linewidth=0.8, alpha=0.6, zorder=1)


ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel(r'$n$',   fontsize=13)
ax.set_xscale('log')
ax.set_ylabel(r'$b_n$', fontsize=13)
ax.set_title(rf'Lanczos coefficients — $N={N}$, BC: {BC}', fontsize=12)
ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()








# %%
from scipy.optimize import curve_fit
from scipy.stats import norm

def gaussian(x, amp, mu_g, sigma):
    return amp * np.exp(-0.5 * ((x - mu_g) / sigma) ** 2)

colors = cm.plasma(np.linspace(0.1, 0.9, len(mu_values)))
fig, ax = plt.subplots(figsize=(9, 5))
sigmas = {}

for (mu, b), color in zip(results.items(), colors):
    label = rf'$\mu = {mu:.2f}$'
    b = np.array(b)
    ratios = np.log(b[:-1] / b[1:])

    counts, bin_edges = np.histogram(ratios, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        p0 = [counts.max(), np.mean(ratios), np.std(ratios)]
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=p0, maxfev=10000)
        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 400)
        ax.plot(x_fit, gaussian(x_fit, *popt), '-', color=color, linewidth=1.8, alpha=0.95, label=label)
        sigmas[mu] = abs(popt[2])
    except RuntimeError:
        pass

ax.axvline(0.0, color='gray', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel(r'$\log(b_n / b_{n+1})$', fontsize=13)
ax.set_ylabel(r'Density', fontsize=13)
ax.set_title(rf'Lanczos log-ratio Gaussian fit — $N={N}$, BC: {BC}', fontsize=12)
ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- sigma vs mu ---
from scipy.interpolate import UnivariateSpline

fig2, ax2 = plt.subplots(figsize=(5, 4))
mu_list  = np.array(list(sigmas.keys()))
sig_list = np.array([sigmas[m] for m in mu_list])

ax2.scatter(mu_list, sig_list, color='black', s=20, zorder=3, edgecolors='none')

spline    = UnivariateSpline(mu_list, sig_list, s=len(mu_list) * (sig_list.std() * 0.4) ** 2)
mu_smooth = np.linspace(mu_list.min(), mu_list.max(), 400)
ax2.plot(mu_smooth, spline(mu_smooth), 'k-', linewidth=1.6, alpha=0.8)

ax2.axvline(2.0, color='red', lw=0.9, ls='--', alpha=0.5, label=r'$\mu_c = 2t$')
ax2.set_xlabel(r'$\mu$', fontsize=13)
ax2.set_ylabel(r'$\sigma$', fontsize=13)
ax2.set_title(r'Gaussian width $\sigma$ vs $\mu$', fontsize=12)
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# %%
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

fig2, ax2 = plt.subplots(figsize=(5, 4))

mu_list  = np.array(list(sigmas.keys()))
sig_list = np.array([sigmas[m] for m in mu_list])

colors_inset = cm.plasma(np.linspace(0.1, 0.9, len(mu_list)))
#ax2.scatter(mu_list, sig_list, color='black', s=20, zorder=3, edgecolors='none')

spline    = UnivariateSpline(mu_list, sig_list, s=len(mu_list) * (sig_list.std() * 0.6) ** 2)
mu_smooth = np.linspace(mu_list.min(), mu_list.max(), 400)
ax2.plot(mu_smooth, spline(mu_smooth), 'k-', linewidth=1.6, alpha=0.8)

ax2.axvline(2.0, color='gray', lw=0.9, ls='--', alpha=0.5, label=r'$\mu_c = 2t$')
ax2.set_xlabel(r'$\mu$', fontsize=13)
ax2.set_ylabel(r'$\sigma$', fontsize=13)
ax2.set_title(r'Gaussian width $\sigma$ vs $\mu$', fontsize=12)
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
