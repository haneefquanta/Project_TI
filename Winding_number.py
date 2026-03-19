import numpy as np
from Kitaev_Kspace import Hk
from Kitaev_Kspace import Hamiltonian_Kspace

N_k = 3       # system size (large for bulk)
t = 1             # hopping
Delta = 1         # pairing
mu = 1.5 


ks, E , vecs = Hamiltonian_Kspace(mu,t, Delta, N_k)  #

print(vecs)
print(vecs.shape)

