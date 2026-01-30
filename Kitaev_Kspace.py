import numpy as np
from Kitaev_realspace import Hamiltonian
import matplotlib.pyplot as plt


N_k = 400         # system size (large for bulk)
t = 1             # hopping
Delta = 1         # pairing
mu = 1.5 

def Hk(eps, delta_k):   
        return np.array([[eps, 1j*delta_k],
                     [-1j*delta_k, -eps]], dtype=complex)


def kitaev_k_space(mu, t, Delta, N_k): 
    ks = np.linspace(-np.pi, np.pi, N_k, endpoint=False) # k = 2*p*n/N_k (N_K is the resolution) 
    energies = np.zeros((N_k,2))    #initiating the energies 
    for i,k in enumerate(ks):       
        eps = -mu - 2*t*np.cos(k)  # eps = -2tcosk-mu 
        delta_k = 2*Delta*np.sin(k) #delta_k = 2del * sin(k)
        vals, vecs = np.linalg.eigh(Hk(eps, delta_k))  # Diagonalisation
        energies[i,:] = np.sort(vals) # each row contain +E and -E  ,#E±​(k)=±(ε(k)**2+Δ(k)**2)**0.5
    return ks, energies


ks, E = kitaev_k_space(mu,t, Delta, N_k) 

plt.figure(figsize=(6,4))
plt.plot(ks, E[:, 0], 'b', label=r'$E_-(k)$')
plt.plot(ks, E[:, 1], 'r', label=r'$E_+(k)$')
plt.axhline(0, color='k', linewidth=0.5)

plt.xlabel(r'$k$')
plt.ylabel(r'Energy $E(k)$')
plt.title(f'Kitaev Chain BdG Spectrum (k-space), mu = {mu}')

plt.legend()
plt.tight_layout()
plt.show()

               





