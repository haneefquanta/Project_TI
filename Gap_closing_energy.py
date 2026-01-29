import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import Hamiltonian

N = 50             # system size (large for bulk)
t = 1.0              # hopping
Delta = 0.6          # pairing
BC = 1



mu_list = np.linspace(-4.0, 4.0, 401)  # sweep chemical potential
gap = np.zeros_like(mu_list)


for i, mu in enumerate(mu_list):
    H = Hamiltonian(N, mu, t, Delta, BC)
    
    # BdG spectrum
    E = np.linalg.eigvalsh(H)  
    # bulk gap = smallest positive Energy Eigen Value
    gap[i] = np.min(np.abs(E))


plt.figure(figsize=(7, 4))
plt.plot(mu_list, gap, lw=2)
plt.axvline( 2*t, ls='--', color='k', alpha=0.6)
plt.axvline(-2*t, ls='--', color='k', alpha=0.6)

plt.xlabel(r'Chemical potential $\mu$')
plt.ylabel('Bulk gap')
plt.title('Bulk gap vs chemical potential (Kitaev chain)')
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()




