# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import sigmaclip


# ═══════════════════════════════════════════════════════════════
# PARAMETERS  ← only this block changes vs Kitaev
# ═══════════════════════════════════════════════════════════════

v = 1                             # intracell hopping (fixed)
w_values = np.linspace(0.1, 4.0, 100)   # sweep intercell hopping w
                                          # transition at w = v = 1.0
N  = 30
BC = 1
w_window = 20    # Krylov window
D = 0          # disorder strength
epsilon = 0         # TRS-breaking imaginary hopping (0 = clean)


# ═══════════════════════════════════════════════════════════════
# SEED OPERATOR  ← changed from Majorana to SSH bond operator
# ═══════════════════════════════════════════════════════════════
# SSH basis: [A1, B1, A2, B2, ..., AN, BN]  (2N sites)
# Bulk bond operator at middle unit cell

O_ssh_bulk = np.zeros((2*N, 2*N), dtype=complex)
mid = N // 2
O_ssh_bulk[2*mid,   2*mid+1] =  1    # A_mid → B_mid
O_ssh_bulk[2*mid+1, 2*mid  ] = -1    # antisymmetric

O_init = O_ssh_bulk


# ═══════════════════════════════════════════════════════════════
# HAMILTONIAN  ← only new function, replaces build_disordered_kitaev
# ═══════════════════════════════════════════════════════════════

def build_ssh(N, v, w, BC=0, W=0.0, epsilon=0.0):
    """
    SSH chain with optional disorder and TRS-breaking imaginary hopping.

    Basis: [A1, B1, A2, B2, ..., AN, BN]  (2N sites, fermionic)

    v       : intracell hopping  (trivial phase when v > w)
    w       : intercell hopping  (topological phase when w > v)
    W       : on-site disorder strength
    epsilon : imaginary hopping — breaks TRS (BDI → class A)
    BC      : 0 = OBC,  1 = PBC
    """
    H = np.zeros((2*N, 2*N), dtype=complex)

    # on-site disorder
    disorder = W * np.random.uniform(-1, 1, 2*N)
    for i in range(2*N):
        H[i, i] = disorder[i]

    for i in range(N):
        # intracell hopping v: A_i <-> B_i
        H[2*i,   2*i+1] =  v + 1j * epsilon
        H[2*i+1, 2*i  ] =  v - 1j * epsilon

        # intercell hopping w: B_i <-> A_{i+1}
        if i < N - 1:
            H[2*i+1, 2*(i+1)  ] =  w + 1j * epsilon
            H[2*(i+1), 2*i+1  ] =  w - 1j * epsilon

    if BC == 1:
        H[2*N-1, 0      ] =  w + 1j * epsilon
        H[0,     2*N-1  ] =  w - 1j * epsilon

    return H


# ═══════════════════════════════════════════════════════════════
# LANCZOS  ← identical to your Kitaev version
# ═══════════════════════════════════════════════════════════════

def lanczos(H, O_init, window=w_window):
    dim          = H.shape[0]
    krylov_basis = []
    a_coeffs     = []
    b_coeffs     = [0.0]

    O_n   = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros((dim, dim), dtype=complex)

    for n in range(window):
        A   = (H @ O_n - O_n @ H) * 1j
        a_n = np.real(np.sum(O_n.conj() * A))
        a_coeffs.append(a_n)

        A = A - a_n * O_n - b_coeffs[n] * O_prev

        for O_k in krylov_basis[-window:]:
            A = A - np.sum(O_k.conj() * A) * O_k
        for O_k in krylov_basis[-window:]:
            A = A - np.sum(O_k.conj() * A) * O_k

        b_next = np.linalg.norm(A)
        if b_next < 1e-3:
            print(f"  Krylov space exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

    return np.array(b_coeffs[1:]), krylov_basis


# ═══════════════════════════════════════════════════════════════
# SINE FIT  ← identical to your Kitaev version
# ═══════════════════════════════════════════════════════════════

def sine_model(n, A, omega, phi, C):
    return A * np.sin(omega * n + phi) + C

def _top_k_fft_freqs(b_centered, k=5):
    n     = len(b_centered)
    freqs = fftfreq(n, d=1)
    power = np.abs(fft(b_centered))
    pos_mask  = freqs > 0
    top_idx   = np.argsort(power[pos_mask])[::-1][:k]
    return 2 * np.pi * freqs[pos_mask][top_idx]

def _refine_omega(b, omega_coarse, search_fraction=0.3):
    n  = np.arange(len(b))
    lo = max(omega_coarse * (1 - search_fraction), 1e-6)
    hi = omega_coarse * (1 + search_fraction)
    def residual(omega):
        X    = np.column_stack([np.sin(omega*n), np.cos(omega*n), np.ones(len(n))])
        beta, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
        return np.sum((b - X @ beta)**2)
    return minimize_scalar(residual, bounds=(lo, hi),
                           method='bounded', options={'xatol': 1e-8}).x

def _analytical_init(b, omega):
    n    = np.arange(len(b))
    X    = np.column_stack([np.sin(omega*n), np.cos(omega*n), np.ones(len(n))])
    beta, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    alpha, beta_cos, C = beta
    A   = np.sqrt(alpha**2 + beta_cos**2)
    phi = np.arctan2(beta_cos, alpha)
    return A, phi, C

def _r_squared(b, b_pred):
    ss_res = np.sum((b - b_pred)**2)
    ss_tot = np.sum((b - b.mean())**2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot

def fit_sine_full(b, k_candidates=5):
    b   = np.array(b, dtype=float)
    n   = np.arange(len(b))
    if len(b) < 4:
        return None

    b_centered = b - b.mean()
    amp_range  = b.max() - b.min()
    candidates_coarse = _top_k_fft_freqs(b_centered, k=k_candidates)

    candidate_omegas = []
    for oc in candidates_coarse:
        try:    candidate_omegas.append(_refine_omega(b, oc))
        except: candidate_omegas.append(oc)

    best_result, best_r2 = None, -np.inf
    for omega_guess in candidate_omegas:
        try:
            A_guess, phi_guess, C_guess = _analytical_init(b, omega_guess)
            p0     = [A_guess, omega_guess, phi_guess, C_guess]
            bounds = (
                [0.0,              omega_guess*0.5, -2*np.pi, b.min()],
                [amp_range + 1e-6, omega_guess*1.5,  2*np.pi, b.max()]
            )
            popt, _ = curve_fit(sine_model, n, b, p0=p0, bounds=bounds,
                                 maxfev=200000, ftol=1e-12, xtol=1e-12)
            r2 = _r_squared(b, sine_model(n, *popt))
            if r2 > best_r2:
                best_r2, best_result = r2, popt
        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return None

    A, omega, phi, C = best_result
    fitted = sine_model(n, A, omega, phi, C)
    return {'params': (A, omega, phi, C),
            'r_squared': best_r2,
            'fitted': fitted,
            'residuals': b - fitted}


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP  ← same structure, w replaces mu
# ═══════════════════════════════════════════════════════════════

results = {}
for w in w_values:
    print(f"Running w = {w:.3f} ...", end="  ")
    H   = build_ssh(N, v, w, BC=BC, W=D, epsilon=epsilon)
    b, _ = lanczos(H, O_init, window=w_window)
    scale = np.sqrt(v**2 + w**2)           # same normalization logic
    results[w] = b / scale
    print(f"done  (K = {len(b)})")

# trim transient  ← identical
for w in w_values:
    results[w] = results[w][3:]


# ═══════════════════════════════════════════════════════════════
# SINE FIT ALL W  ← identical
# ═══════════════════════════════════════════════════════════════

fit_results = {}
for w, b in results.items():
    fit_results[w] = fit_sine_full(np.array(b))

w_arr   = np.array(sorted(fit_results.keys()))
phi_arr = np.array([
    fit_results[w]['params'][2] if fit_results[w] is not None else np.nan
    for w in w_arr
])
amp_arr = np.array([
    fit_results[w]['params'][0] if fit_results[w] is not None else np.nan
    for w in w_arr
])

valid    = ~np.isnan(phi_arr)
phi_plot = phi_arr.copy()
phi_plot[valid] -= phi_arr[valid][0]    # first point = reference 0


# ═══════════════════════════════════════════════════════════════
# COLOUR MAP  ← same scheme, trivial/topo swapped for SSH
# (topological: w > v=1,  trivial: w < v=1)
# ═══════════════════════════════════════════════════════════════

critical_band = (0.85, 1.15)

topo_ws    = [w for w in results.keys() if w > critical_band[1]]   # w > v
trivial_ws = [w for w in results.keys() if w < critical_band[0]]   # w < v

topo_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo', ['#2d006b', '#6a00cc', '#aa44ff', '#dd99ff'])
topo_norm = mcolors.Normalize(
    vmin=min(topo_ws) if topo_ws else 0,
    vmax=max(topo_ws) if topo_ws else 1)

trivial_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trivial', ['#fff176', '#ffd600', '#f9a825', '#6b4000'])
trivial_norm = mcolors.Normalize(
    vmin=min(trivial_ws) if trivial_ws else 0,
    vmax=max(trivial_ws) if trivial_ws else 1)

def get_color(w):
    if critical_band[0] <= w <= critical_band[1]:
        return 'black'
    elif w > critical_band[1]:                      # topological: w > v
        return topo_cmap(topo_norm(w))
    else:                                            # trivial: w < v
        return trivial_cmap(trivial_norm(w))

legend_entries = [
    mpatches.Patch(color='#6a00cc', label=r'Topological ($w > v$) — dark→light violet'),
    mpatches.Patch(color='black',   label=r'Critical region ($w \approx v$)'),
    mpatches.Patch(color='#ffd600', label=r'Trivial ($w < v$) — light→dark yellow'),
]



fig_bn, ax_bn = plt.subplots(figsize=(10, 5))

for w, b in results.items():
    color       = get_color(w)
    is_critical = critical_band[0] <= w <= critical_band[1]
    lw          = 1.6 if is_critical else 0.7
    alpha       = 0.95 if is_critical else 0.65
    n_b         = np.arange(1, len(b) + 1)
    ax_bn.scatter(n_b, b, color=color, s=3, alpha=alpha, zorder=2)
    ax_bn.plot(   n_b, b, '-', color=color, linewidth=lw, alpha=alpha, zorder=1)

ax_bn.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)

legend_entries_bn = [
    mpatches.Patch(color='#ffd600', label=r'Trivial ($w < v$) — light→dark yellow'),
    mpatches.Patch(color='black',   label=r'Critical region ($w \approx v$)'),
    mpatches.Patch(color='#6a00cc', label=r'Topological ($w > v$) — dark→light violet'),
]
ax_bn.legend(handles=legend_entries_bn, fontsize=10, loc='upper right', framealpha=0.88)

ax_bn.set_xlabel(r'$n$',                        fontsize=13)
ax_bn.set_ylabel(r'$\mathrm{normalised}\ b_n$', fontsize=13)
ax_bn.set_title(rf'SSH chain: Lanczos coefficients $b_n$ — $N={N}$, BC: {BC}', fontsize=12)
ax_bn.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bn_vs_n_SSH.png", dpi=150, bbox_inches='tight')
plt.show()
# ═══════════════════════════════════════════════════════════════
# PLOT  ← identical to your Kitaev plot, axis labels updated
# ═══════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel 1: Phase ────────────────────────────────────────────
ax1.plot(w_arr[valid], phi_plot[valid],
         '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for w, phi in zip(w_arr[valid], phi_plot[valid]):
    ax1.scatter(w, phi, color=get_color(w), s=20, zorder=3)

ax1.axvline(v,   color='red',  ls='--', lw=1.2, label=r'$w = v$ (transition)')
ax1.axhline(0,   color='gray', ls=':',  lw=1.0)
ax1.axvspan(w_arr.min(), v, alpha=0.06, color='yellow', label='Trivial')
ax1.axvspan(v, w_arr.max(),             alpha=0.06, color='purple', label='Topological')

pi_ticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticks(pi_ticks)
ax1.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=10)
ax1.set_xlabel(r'$w$ (intercell hopping)', fontsize=13)
ax1.set_ylabel(r'$\Delta\phi$ (rad)',       fontsize=13)
ax1.set_title(rf'SSH chain: Phase $\Delta\phi$ vs $w$ — $N={N}$ sites', fontsize=12)
ax1.legend(handles=legend_entries, fontsize=8, loc='upper right', framealpha=0.88)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Amplitude ────────────────────────────────────────
ax2.plot(w_arr[valid], amp_arr[valid],
         '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for w, amp in zip(w_arr[valid], amp_arr[valid]):
    ax2.scatter(w, amp, color=get_color(w), s=20, zorder=3)

ax2.axvline(v,   color='red',  ls='--', lw=1.2)
ax2.axhline(0,   color='gray', ls=':',  lw=1.0)
ax2.axvspan(w_arr.min(), v, alpha=0.06, color='yellow')
ax2.axvspan(v, w_arr.max(),             alpha=0.06, color='purple')

ax2.set_xlabel(r'$w$ (intercell hopping)', fontsize=13)
ax2.set_ylabel(r'$A$',                     fontsize=13)
ax2.set_title(rf'SSH chain: Amplitude $A$ vs $w$ — $N={N}$ sites', fontsize=12)
ax2.legend(handles=legend_entries, fontsize=8, loc='upper right', framealpha=0.88)
ax2.grid(True, alpha=0.3)

fig.suptitle(rf'SSH chain — Krylov sine fit diagnostics, $v={v}$, $N={N}$ sites',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("phi_and_amplitude_vs_w_SSH.png", dpi=150, bbox_inches='tight')
plt.show()

# %%