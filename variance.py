# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
import matplotlib.cm as cm

# ─── Parameters ─────────────────────────────────────────────────────────────
mu_values = [1,1.5,1.9,2,2.1,3]
t       = 1
delta   = 1
N       = 12
BC      = 0

O_init = np.zeros((2*N, 2*N), dtype=complex)
O_init[0,    2*N-1] = -1
O_init[2*N-1, 0   ] =  1

w = N * 2 * (N-1) + 1   # Krylov window

# ─── Lanczos ────────────────────────────────────────────────────────────────
def lanczos(H, O_init, window=w):
    dim          = H.shape[0]
    krylov_basis = []
    a_coeffs     = []
    b_coeffs     = [0.0]

    O_n    = O_init / np.linalg.norm(O_init, 'fro')
    O_prev = np.zeros_like(O_n)
    krylov_basis.append(O_n.copy())

    for n in range(window):
        A   = (H @ O_n - O_n @ H) * 1j
        a_n = np.real(np.sum(O_n.conj() * A))          # Frobenius ⟨O_n|A⟩
        a_coeffs.append(a_n)
        A   = A - a_n * O_n - b_coeffs[n] * O_prev

        # double reorthogonalisation
        for O_k in krylov_basis[-window:]:
            A -= np.sum(O_k.conj() * A) * O_k
        for O_k in krylov_basis[-window:]:
            A -= np.sum(O_k.conj() * A) * O_k

        b_next = np.linalg.norm(A, 'fro')
        if b_next < 1e-3:
            print(f"Krylov space exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

    b_coeffs = b_coeffs[1:]
    return np.array(b_coeffs), krylov_basis


# ─── Run Lanczos ────────────────────────────────────────────────────────────
results    = {}
results_kb = {}

for mu in mu_values:
    print(f"Running μ = {mu} ...", end="  ")
    H_m    = build_HM_blocks(N, mu, t, delta, BC, -1)
    b, K_b = lanczos(H_m, O_init)
    results[mu]    = b
    results_kb[mu] = K_b
    print(f"done  (K = {len(b)})")


# ─── Eigen Basis Projection ──────────────────────────────────────────────────
def diagonalize_H(H):
    """
    Diagonalise H, handling Hermitian, skew-Hermitian (Majorana BdG), or general.

    Majorana BdG convention: H_m is real antisymmetric  →  skew-Hermitian.
    Eigenvalues are purely imaginary ±iε_k; physical energies = imag(E).
    """
    if np.allclose(H, H.conj().T, atol=1e-8):
        E, U = np.linalg.eigh(H)          # real eigenvalues, orthonormal U
    elif np.allclose(H, -H.conj().T, atol=1e-8):
        # skew-Hermitian: i H is Hermitian
        eps, U = np.linalg.eigh(1j * H)   # real ε_k
        E = -1j * eps                      # E_k = -i ε_k  (purely imaginary)
    else:
        E, U = np.linalg.eig(H)
        idx  = np.argsort(np.real(E))
        E, U = E[idx], U[:, idx]
    return E, U


def eigen_projection(H, O_init):
    """
    Project the normalised O_init onto the eigenbasis of H.

    Returns
    -------
    E       : (2N,)  eigenvalues
    O_eig   : (2N,2N) O in eigenbasis, O_eig[m,n] = <m|O|n>
    weights : (2N,2N) spectral weights |O_eig[m,n]|²
    freqs   : (2N,2N) Liouvillian frequencies ω_{mn} = Re(E_m - E_n)
              For real-eigenvalue H: ω_{mn} = E_m - E_n
              For imaginary-eigenvalue H (skew-Hermitian): ω_{mn} = Im(E_m - E_n)
    """
    O_norm  = O_init / np.linalg.norm(O_init, 'fro')
    E, U    = diagonalize_H(H)

    O_eig   = U.conj().T @ O_norm @ U

    weights = np.abs(O_eig) ** 2

    # Pick out the "real" frequency axis regardless of convention
    diff    = E[:, None] - E[None, :]
    if np.allclose(np.real(E), 0, atol=1e-8):      # purely imaginary eigenvalues
        freqs = np.imag(diff)
    else:
        freqs = np.real(diff)

    return E, O_eig, weights, freqs


def spectral_density(weights, freqs, n_bins=300, eta_factor=0.5):
    """
    Lorentzian-broadened spectral function
        ρ(ω) = Σ_{mn} w_{mn}  η/π / [(ω - ω_{mn})² + η²]
    """
    w_flat = weights.flatten()
    f_flat = freqs.flatten()

    mask   = w_flat > 1e-12
    w_flat, f_flat = w_flat[mask], f_flat[mask]

    f_range  = f_flat.max() - f_flat.min()
    eta      = eta_factor * f_range / n_bins
    omega_g  = np.linspace(f_flat.min() - 2*eta, f_flat.max() + 2*eta, 1000)

    rho = np.sum(
        w_flat[None, :] * eta / ((omega_g[:, None] - f_flat[None, :])**2 + eta**2),
        axis=1
    ) / np.pi

    return omega_g, rho, f_flat, w_flat


def autocorrelation_spectral(weights, freqs, t_array):
    """
    Survival amplitude via spectral decomposition:
        C(t) = Σ_{mn} w_{mn} exp(i ω_{mn} t)
    """
    w_flat = weights.flatten()
    f_flat = freqs.flatten()
    C = np.array([np.dot(w_flat, np.exp(1j * f_flat * t)) for t in t_array])
    return C


def krylov_propagation(b_coeffs, t_array):
    """
    Propagate |ψ(t)⟩ = exp(-i T t) |K_0⟩  on the tridiagonal Krylov chain.

    Returns
    -------
    K_t  : Krylov complexity Σ_n n |φ_n(t)|²
    C_t  : survival amplitude φ_0(t) = ⟨K_0|ψ(t)⟩
    """
    dim  = len(b_coeffs) + 1
    T    = np.zeros((dim, dim))
    for i, b in enumerate(b_coeffs):
        T[i,   i+1] = b
        T[i+1, i  ] = b

    evals, evecs = np.linalg.eigh(T)
    psi0         = np.zeros(dim); psi0[0] = 1.0
    coeffs0      = evecs.T @ psi0       # project onto T's eigenvectors

    n_vals = np.arange(dim)
    K_t, C_t = [], []

    for tau in t_array:
        phases = np.exp(-1j * evals * tau)
        psi_t  = evecs @ (phases * coeffs0)
        probs  = np.abs(psi_t) ** 2
        K_t.append(np.dot(n_vals, probs))
        C_t.append(psi_t[0])

    return np.array(K_t), np.array(C_t)


# ─── Run eigen projection ────────────────────────────────────────────────────
t_max   = 30
t_array = np.linspace(0, t_max, 600)

eigen_data = {}
for mu in mu_values:
    H_m = build_HM_blocks(N, mu, t, delta, BC, -1)
    E, O_eig, weights, freqs = eigen_projection(H_m, O_init)
    omega_g, rho, f_flat, w_flat = spectral_density(weights, freqs)
    C_spec  = autocorrelation_spectral(weights, freqs, t_array)
    K_t, C_kry = krylov_propagation(results[mu], t_array)

    eigen_data[mu] = dict(
        E=E, O_eig=O_eig, weights=weights, freqs=freqs,
        omega_g=omega_g, rho=rho, f_flat=f_flat, w_flat=w_flat,
        C_spec=C_spec, K_t=K_t, C_kry=C_kry
    )


# ─── Plotting ────────────────────────────────────────────────────────────────
colors = cm.plasma(np.linspace(0.15, 0.85, len(mu_values)))

# ── 1. Original b_n plot ─────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))
for (mu, b), color in zip(results.items(), colors):
    label = rf'$\mu = {mu}$' + (' ← critical' if mu == 2.0 else '')
    n_b   = np.arange(1, len(b) + 1)
    ax1.scatter(n_b, b, color=color, s=4, alpha=0.6, label=label, zorder=2)
    ax1.plot(n_b, b, '-', color=color, linewidth=0.8, alpha=0.6, zorder=1)
ax1.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax1.set_xlabel(r'$n$', fontsize=13)
ax1.set_ylabel(r'$b_n$', fontsize=13)
ax1.set_title(rf'Lanczos coefficients — $N={N}$, BC: {BC}', fontsize=12)
ax1.legend(fontsize=9, framealpha=0.85)
ax1.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ── 2. Spectral density ρ(ω) ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, len(mu_values), figsize=(5*len(mu_values), 4),
                            sharey=False)
if len(mu_values) == 1:
    axes2 = [axes2]
for ax, (mu, color) in zip(axes2, zip(mu_values, colors)):
    d = eigen_data[mu]
    ax.fill_between(d['omega_g'], d['rho'], alpha=0.35, color=color)
    ax.plot(d['omega_g'], d['rho'], color=color, lw=1)
    ax.scatter(d['f_flat'], np.zeros_like(d['f_flat']),
               s=d['w_flat']*600, color=color, alpha=0.5,
               zorder=3, label='Spectral poles')
    ax.set_xlabel(r'$\omega_{mn} = E_m - E_n$', fontsize=12)
    ax.set_ylabel(r'$\rho(\omega)$', fontsize=12)
    ax.set_title(rf'Spectral density, $\mu={mu}$', fontsize=11)
    ax.grid(True, alpha=0.3)
plt.suptitle('Liouvillian spectral density (Majorana basis)', fontsize=12)
plt.tight_layout(); plt.show()

# ── 3. Spectral weight matrix |Ô_{mn}|² ─────────────────────────────────────
fig3, axes3 = plt.subplots(1, len(mu_values), figsize=(5*len(mu_values), 4))
if len(mu_values) == 1:
    axes3 = [axes3]
for ax, mu in zip(axes3, mu_values):
    d = eigen_data[mu]
    im = ax.imshow(np.log10(d['weights'] + 1e-14), aspect='auto',
                   cmap='inferno', origin='lower')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}|{\hat O}_{mn}|^2$')
    ax.set_xlabel('n (eigenindex)', fontsize=11)
    ax.set_ylabel('m (eigenindex)', fontsize=11)
    ax.set_title(rf'Spectral weight matrix, $\mu={mu}$', fontsize=11)
plt.tight_layout(); plt.show()

# ── 4. Krylov complexity K(t) ────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 4))
for (mu, color) in zip(mu_values, colors):
    ax4.plot(t_array, eigen_data[mu]['K_t'], color=color,
             lw=1.4, label=rf'$\mu={mu}$')
ax4.set_xlabel(r'$t$', fontsize=13)
ax4.set_ylabel(r'$K(t)$', fontsize=13)
ax4.set_title(rf'Krylov complexity — $N={N}$, BC: {BC}', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ── 5. Survival amplitude |C(t)|²  — spectral vs Krylov chain ───────────────
fig5, axes5 = plt.subplots(1, len(mu_values), figsize=(5*len(mu_values), 4),
                            sharey=True)
if len(mu_values) == 1:
    axes5 = [axes5]
for ax, (mu, color) in zip(axes5, zip(mu_values, colors)):
    d = eigen_data[mu]
    ax.plot(t_array, np.abs(d['C_spec'])**2,
            color=color,  lw=1.6, label='Spectral decomp.')
    ax.plot(t_array, np.abs(d['C_kry'])**2,
            color='k', lw=0.9, ls='--', alpha=0.7, label='Krylov chain')
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$|C(t)|^2$', fontsize=12)
    ax.set_title(rf'Survival amplitude, $\mu={mu}$', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Spectral vs Krylov-chain survival amplitude', fontsize=12)
plt.tight_layout(); plt.show()

# ── 6. Eigenvalue spectrum ────────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(9, 3))
for mu, color in zip(mu_values, colors):
    E = eigen_data[mu]['E']
    # plot on complex plane (x = Re E, y = Im E)
    ax6.scatter(np.real(E), np.imag(E), color=color, s=18, alpha=0.75,
                label=rf'$\mu={mu}$')
ax6.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax6.axvline(0, color='gray', lw=0.6, ls='--', alpha=0.5)
ax6.set_xlabel(r'$\mathrm{Re}(E)$', fontsize=12)
ax6.set_ylabel(r'$\mathrm{Im}(E)$', fontsize=12)
ax6.set_title(rf'Eigenvalue spectrum of $H_M$ — $N={N}$, BC: {BC}', fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# ── 7. Variance of b_n ────────────────────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(7, 4))
mu_list  = list(results.keys())
var_list = [np.var(b) for b in results.values()]
ax7.scatter(mu_list, var_list, color='steelblue', s=60, zorder=2)
ax7.plot(mu_list, var_list, '-', color='steelblue', lw=1.2, alpha=0.7)
ax7.axvline(2.0, color='red', lw=1.2, ls='--', alpha=0.8, label=r'$\mu_c = 2t$')
ax7.set_xlabel(r'$\mu$', fontsize=13)
ax7.set_ylabel(r'$\mathrm{Var}(b_n)$', fontsize=13)
ax7.set_title(rf'Variance of Lanczos coefficients — $N={N}$, BC: {BC}', fontsize=12)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
# %%