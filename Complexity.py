# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
from two_body_basis import lanczos

mu = 3
N = 20
BC = 1
O_init = np.zeros((2*N,2*N), dtype=complex)
site = 0    # mi
H_m =  build_HM_blocks(N, mu, 1, 1, BC,-1)
O_init[0,N] = -1
O_init[N,0] = 1
b_coeffs = lanczos(H_m, O_init)

D = len(b_coeffs) + 1
sub_diag   = np.diag(b_coeffs, k=-1)
super_diag = np.diag(b_coeffs, k=1)
M = 1j * (sub_diag + super_diag)   # from i·φ̇ = b_n φ_{n-1} + b_{n+1} φ_{n+1}

phi0 = np.zeros(D, dtype=complex)
phi0[0] = 1.0

T       = 1500
t_steps = 150
t_values = np.linspace(0, T, t_steps)


#idphi_n= bn*phi_n-1 + bn+1*phi_n+1 

#  RHS for solve_ivp 
def rhs(t, phi):
    return M @ phi

#  Solve 
sol = solve_ivp(
    rhs,
    t_span = (0, T),
    y0     = phi0,
    t_eval = t_values,
    method = 'RK45',
    rtol   = 1e-10,
    atol   = 1e-12
)

if not sol.success:
    raise RuntimeError(f"Solver failed: {sol.message}")

# sol.y has shape (D, T) → transpose to (T, D)
Phi = sol.y.T

n             = np.arange(D)
probabilities = np.abs(Phi)**2                    # shape (T, D)
Complexity    = np.sum(n * probabilities, axis=1) # shape (T,)
#Complexity = Complexity/(Complexity.max())


plt.figure(figsize=(8, 5))
plt.plot(sol.t, Complexity, color='royalblue', linewidth=2)
plt.xlabel("t", fontsize=13)
plt.ylabel("K(t)", fontsize=13)
plt.title(f"Krylov Complexity (solve_ivp) | μ = {mu}, N = {N}, BC = {BC}", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("krylov_complexity.png", dpi=300, bbox_inches='tight')
plt.show()



# %%
plt.imshow(probabilities.T, aspect='auto', origin='lower', extent=[0, T, 0, D])
plt.colorbar(label="|φ_n(t)|^2")
plt.xlabel("Time")
plt.ylabel("Krylov Node (n)")
plt.title("Operator Spreading in Krylov Space")
# %%