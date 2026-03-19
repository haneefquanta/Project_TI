# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
from Krylov_basis import lanczos

N = 50
BC = 1
O_init = np.zeros(2*N, dtype=complex)
O_init[0] = 1.0
mu_values = np.linspace(-5, -1, 30)
K_peak = []

for mu in mu_values:
    H_m = build_HM_blocks(N, mu, 1, 1, BC, parity=-1)
    b_coeffs = lanczos(H_m, O_init)
    D = len(b_coeffs) + 1
    sub_diag   = np.diag(b_coeffs, k=-1)
    super_diag = np.diag(b_coeffs, k=1)
    M = 1j * (sub_diag + super_diag)   # from i·φ̇ = b_n φ_{n-1} + b_{n+1} φ_{n+1}

    phi0 = np.zeros(D, dtype=complex)
    phi0[0] = 1.0
    T       = 2000
    t_steps = 100
    t_values = np.linspace(0, T, t_steps)

    # ── RHS for solve_ivp ─────────────────────────────────────────
    def rhs(t, phi):
        return M @ phi

    # ── Solve ─────────────────────────────────────────────────────
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
    # ── Complexity ────────────────────────────────────────────────
    n             = np.arange(D)
    probabilities = np.abs(Phi)**2                    # shape (T, D)
    Complexity    = np.sum(n * probabilities, axis=1) 
    K_peak.append(np.max(Complexity))


plt.plot(mu_values, K_peak)
plt.axvline(x=2, color='red', linestyle='--', label='Critical point |μ|=2J')
plt.xlabel("μ")
plt.ylabel("K_peak")
plt.title("Peak Krylov Complexity vs μ")
plt.show()

# %%
