import numpy as np
import matplotlib.pyplot as plt



#N--> Lattice Sites , mu --> on site potential, BC - Boundary condition
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

    H = np.block([[h,P],
                 [P.conj().T , -h.T  ]])

    return H 





        


        
     
     
