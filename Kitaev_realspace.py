# %%
import numpy as np
import matplotlib.pyplot as plt



#N > Lattice Sites , mu --> on site potential, BC - Boundary condition
def Hamiltonian(N,mu,t,p,BC):
    #single particle Hamiltonian
    h = np.zeros((N,N),dtype=complex)
    #on site chemical potential 
    for i in range(N):
        h[i,i] = -mu

    #hopping terms     
    for i in range(N-1):
        h[i,i+1] = -t
        h[i+1,i] = -t

    #pairing
    P = np.zeros((N,N),dtype=complex)
    for i in range(N-1):
        P[i,i+1] = p
        P[i+1,i] = -p  #negative becasue CiCj anticommutes

    if BC == 1:  #if PBC
        h[N-1,0] = -t
        h[0,N-1] = -t
        P[N-1,0] = p
        P[0,N-1] = -p 

    H_real = np.block([[h,P],  # Basi is (Ci Ci+)
                 [P.conj().T , -h.T  ]]) #for real P, P.conj().T is = -P

    return H_real 

def transformation(H,N):

    I = np.eye(N,dtype=complex)
    S = np.sqrt(1/2) *  np.block([[I,1j*I], #basis is ( Rj,a Rj,b) 
                 [I,-1j*I]])
    S_inv = S.conj().T
    H_M = S_inv @ H @ S

    return H_M





def build_HM_blocks(N, mu, t, delta, BC, parity):
    
    H_AB = np.zeros((N, N), dtype=complex)

    # On-site (diagonal of H_AB)
    for j in range(N):
        H_AB[j, j] = -mu/2  

    
    for j in range(N-1):
        H_AB[j,   j+1] = (delta - t)/2    # A_j  → B_{j+1}
        H_AB[j+1, j  ] = -(t + delta)/2 # A_{j+1} → B_j (from H_BA = -H_AB^T)

    # PBC corner terms
    if BC == 1:  
        H_AB[N-1, 0] = -1*parity * (delta - t)/2
        H_AB[0, N-1] = -1*parity * -(t + delta)/2 

    # Full H_M
    H_m = np.block([[np.zeros((N,N)),  H_AB          ],
                    [-H_AB.T,          np.zeros((N,N))]])


  

    return H_m

 






        


        
     
     

# %%
