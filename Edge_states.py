import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import Hamiltonian

N = 20      # system size (large for bulk)
t = 1             # hopping
Delta = 1         # pairing
BC = 0   #OBC
mu = 0.5 #Topological Phase For MZM detection

H = Hamiltonian(N, mu, t, Delta, BC)
evals, evecs = np.linalg.eigh(H)

mzm_indices = np.argsort(np.abs(evals))[:2]  
print(evals[mzm_indices])
W1 = evecs[:, mzm_indices[0]] 
W2 = evecs[:, mzm_indices[1]]
v1 = W1[:N]
v2 = W2[:N]
u1 = W1[N:]   
u2 = W2[N:]
prob_density_1 = np.abs(u1)**2 + np.abs(v1)**2
prob_density_2 = np.abs(u2)**2 + np.abs(v2)**2
sites = np.arange(1, N + 1)


plt.figure(figsize=(8, 4))
plt.bar(sites, prob_density_1, color='crimson', alpha=0.7)
plt.xlabel("Site Index (j)")
plt.ylabel("Probability Density_1 $|\psi|^2$")
plt.title("Majorana Zero Mode Localization")
plt.show()

plt.figure(figsize=(8, 4))
plt.bar(sites, prob_density_2, color='crimson', alpha=0.7)
plt.xlabel("Site Index (j)")
plt.ylabel("Probability Density_2 $|\psi|^2$")
plt.title("Majorana Zero Mode Localization")
plt.show()

