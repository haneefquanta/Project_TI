# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit          # ← add here at the top



mu_values =np.linspace(-4,4,100)

#mu_values = [0.2,2,2.2,2.3,2.4,2.5,2.6,2.7,2.8 ,2.9 ,3,4]
t= 1
delta = 1

N = 30
BC = 0
O_init = np.zeros((2*N,2*N), dtype=complex)
site = 0    # mi
w = 30
#w =50
D = 0
epsilon = 0

O_first = np.zeros((2*N,2*N), dtype=complex)
O_first[0,N] = 1
O_first[N,0] = -1


O_edge = np.zeros((2*N,2*N), dtype=complex)
O_edge[0,2*N - 1] = -1
O_edge[2*N -1 ,0] = 1


# bulk seed operator — no edge overlap
O_bulk = np.zeros((2*N, 2*N), dtype=complex)
mid = N // 2 + 1
O_bulk[mid, mid + N] = -1
O_bulk[mid + N, mid] =  1

O_init = O_bulk
#O_init = O_first
#O_init =+ O_edge




def lanczos(H, O_init,window=w):
    dim    = H.shape[0]
    krylov_basis = []
    a_coeffs = []
    b_coeffs = [0.0]

    

    O_n = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros((dim, dim))

    for n in range(window):
        
        

        A = (H @ O_n - O_n @ H) * 1j
        
        
        a_n = np.real(O_n.conj() @ A)         # ← fixed inner product
        a_coeffs.append(a_n)

        A = A - a_n * O_n - b_coeffs[n] * O_prev
        
        #   reorthogonalization 
        for O_k in krylov_basis[-window:]: 
            A = A - np.sum(O_k.conj() * A) * O_k
        #for O_k in krylov_basis[-window:]:   
           # A = A - np.sum(O_k.conj() * A) * O_k
        
        
        b_next = np.linalg.norm(A)

        if b_next < 1e-3:
            print(f"Krylov space exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

       

    b_coeffs = b_coeffs[1:]
    return np.array(b_coeffs) , krylov_basis
def build_disordered_kitaev(N, mu, t, delta, BC,parity,epsilon, W):
 
    # generate random chemical potential for each site
    mu_i = mu + W * np.random.uniform(-1, 1, N)
    
    H_AB = np.zeros((N, N), dtype=complex)

    # On-site (diagonal of H_AB)
    for j in range(N):
        H_AB[j, j] = -mu_i[j]/2  

    
    for j in range(N-1):
        H_AB[j,   j+1] = (delta - t)/2  + 1j * epsilon  # A_j  → B_{j+1}
        H_AB[j+1, j  ] = -(t + delta)/2  - 1j * epsilon # A_{j+1} → B_j (from H_BA = -H_AB^T)

    # PBC corner terms
    if BC == 1:  
        H_AB[N-1, 0] = -1*parity * (delta - t)/2 + 1j * epsilon
        H_AB[0, N-1] = -1*parity * -(t + delta)/2 - 1j * epsilon

    # Full H_M
    H_m = np.block([[np.zeros((N,N)),  H_AB          ],
                    [-H_AB.T,          np.zeros((N,N))]])


  

    return H_m




results = {}
results_kb = {}
for mu in mu_values:
    print(f"Running μ = {mu} ...", end="  ")
    H_m = build_disordered_kitaev(N, mu, t, delta, BC,1, epsilon, D)
    b ,K_b  = lanczos(H_m, O_init) 
    scale = np.sqrt(((mu/2) + t)**2 + delta**2)
    results[mu] = b/scale
    results_kb[mu] = K_b  
    print(results[mu])                      
    print(f"done  (K = {len(b)})")
   


import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 5))

critical_band = (1.8, 3)

topo_mus    = [mu for mu in results.keys() if mu < critical_band[0]]
trivial_mus = [mu for mu in results.keys() if mu > critical_band[1]]

# Topological: dark red (mu=0) → light salmon/pink (mu→1.8)
topo_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo', ['#6b0000', '#cc0000', '#ff4444', '#ff9999']
)
topo_norm = mcolors.Normalize(
    vmin=min(topo_mus) if topo_mus else 0,
    vmax=max(topo_mus) if topo_mus else 1
)

# Trivial: light green (mu just above 2) → dark green (mu=10)
trivial_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trivial', ['#aaffaa', '#44cc44', '#1a8c1a', '#004d00']
)
trivial_norm = mcolors.Normalize(
    vmin=min(trivial_mus) if trivial_mus else 0,
    vmax=max(trivial_mus) if trivial_mus else 1
)

def get_color(mu):
    if critical_band[0] <= mu <= critical_band[1]:
        return 'black'
    elif mu < critical_band[0]:
        return topo_cmap(topo_norm(mu))
    else:
        return trivial_cmap(trivial_norm(mu))

for mu, b in results.items():
    color = get_color(mu)
    is_critical = critical_band[0] <= mu <= critical_band[1]
    lw    = 1.6 if is_critical else 0.7
    alpha = 0.95 if is_critical else 0.65
    n_b   = np.arange(1, len(b) + 1)
    ax.scatter(n_b, b, color=color, s=3, alpha=alpha, zorder=2)
    ax.plot(   n_b, b, '-', color=color, linewidth=lw, alpha=alpha, zorder=1)

ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)

# ── Custom legend ──────────────────────────────────────────────
legend_entries = [
    mpatches.Patch(color='#cc0000', label=r'Topological phase  ($\mu < 2t$)  — dark→light red'),
    mpatches.Patch(color='black',   label=r'Critical region  ($1.8 \leq \mu \leq 2.2$)'),
    mpatches.Patch(color='#1a8c1a', label=r'Trivial phase  ($\mu > 2t$)  — light→dark green'),
]
ax.legend(handles=legend_entries, fontsize=10, loc='upper right', framealpha=0.88)

ax.set_xlabel(r'$n$', fontsize=13)
ax.set_ylabel(r'$\mathrm{normalised}\ b_n$', fontsize=13)
ax.set_title(rf'Lanczos coefficients — $N={N}$, BC: {BC}', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()










# ═══════════════════════════════════════════════════════════════
#.      PHI CALCULATION
# ═══════════════════════════════════════════════════════════════


for mu in mu_values:
    results[mu] = results[mu][3:]


import numpy as np
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import sigmaclip
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════════
# 1. MODEL
# ═══════════════════════════════════════════════════════════════

def sine_model(n, A, omega, phi, C):
    return A * np.sin(omega * n + phi) + C


# ═══════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════

def _top_k_fft_freqs(b_centered, k=5):
    """Return top-k dominant angular frequencies from FFT power spectrum."""
    n = len(b_centered)
    freqs = fftfreq(n, d=1)
    power = np.abs(fft(b_centered))

    pos_mask  = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]

    top_idx    = np.argsort(pos_power)[::-1][:k]
    top_omegas = 2 * np.pi * pos_freqs[top_idx]
    return top_omegas


def _refine_omega(b, omega_coarse, search_fraction=0.3):
    """
    Sub-bin frequency refinement via scalar minimisation of linear-LS residual.
    Overcomes the 1/N FFT resolution limit.
    """
    n  = np.arange(len(b))
    lo = max(omega_coarse * (1 - search_fraction), 1e-6)
    hi = omega_coarse * (1 + search_fraction)

    def residual(omega):
        S  = np.sin(omega * n)
        Co = np.cos(omega * n)
        X  = np.column_stack([S, Co, np.ones(len(n))])
        beta, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
        return np.sum((b - X @ beta) ** 2)

    result = minimize_scalar(residual, bounds=(lo, hi),
                             method='bounded', options={'xatol': 1e-8})
    return result.x


def _analytical_init(b, omega):
    """
    Analytically solve for A, phi, C given fixed omega via sine-cosine regression.
    Gives curve_fit an exact warm start instead of a blind phase=0 guess.
    """
    n  = np.arange(len(b))
    S  = np.sin(omega * n)
    Co = np.cos(omega * n)
    X  = np.column_stack([S, Co, np.ones(len(n))])
    beta, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    alpha, beta_cos, C = beta
    A   = np.sqrt(alpha**2 + beta_cos**2)
    phi = np.arctan2(beta_cos, alpha)
    return A, phi, C


def _r_squared(b, b_pred):
    """Coefficient of determination R²."""
    ss_res = np.sum((b - b_pred) ** 2)
    ss_tot = np.sum((b - b.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


# ═══════════════════════════════════════════════════════════════
# 3. MAIN FITTING FUNCTION
# ═══════════════════════════════════════════════════════════════

def fit_sine_full(b, k_candidates=5, remove_outliers=False):
    """
    Robustly fit  y(n) = A·sin(ω·n + φ) + C  to signal b.

    Returns dict: {'params': (A, omega, phi, C),
                   'r_squared': float,
                   'fitted': array,
                   'residuals': array}
    Returns None if all attempts fail.
    """
    b = np.array(b, dtype=float)
    n = np.arange(len(b))

    if len(b) < 4:
        raise ValueError("Need at least 4 points to fit a sine wave.")

    # ── Optional outlier removal ─────────────────────────────────────────
    if remove_outliers:
        _, lo, hi = sigmaclip(b, low=3.0, high=3.0)
        mask  = (b >= lo) & (b <= hi)
        n_fit = n[mask]
        b_fit = b[mask]
    else:
        n_fit, b_fit = n, b

    b_centered = b_fit - b_fit.mean()
    amp_range  = b_fit.max() - b_fit.min()

    # ── Stage 1: top-k FFT candidates ────────────────────────────────────
    candidate_omegas_coarse = _top_k_fft_freqs(b_centered, k=k_candidates)

    # ── Stage 2: sub-bin refinement for each candidate ───────────────────
    candidate_omegas = []
    for omega_c in candidate_omegas_coarse:
        try:
            candidate_omegas.append(_refine_omega(b_fit, omega_c))
        except Exception:
            candidate_omegas.append(omega_c)

    # ── Stage 3: multi-start curve_fit, keep best R² ─────────────────────
    best_result = None
    best_r2     = -np.inf

    for omega_guess in candidate_omegas:
        try:
            A_guess, phi_guess, C_guess = _analytical_init(b_fit, omega_guess)

            p0 = [A_guess, omega_guess, phi_guess, C_guess]
            bounds = (
                [0.0,               omega_guess * 0.5, -2 * np.pi, b_fit.min()],
                [amp_range + 1e-6,  omega_guess * 1.5,  2 * np.pi, b_fit.max()]
            )

            popt, _ = curve_fit(
                sine_model, n_fit, b_fit,
                p0=p0, bounds=bounds,
                maxfev=200000, ftol=1e-12, xtol=1e-12,
            )

            fitted = sine_model(n, *popt)
            r2     = _r_squared(b, fitted)

            if r2 > best_r2:
                best_r2     = r2
                best_result = popt

        except (RuntimeError, ValueError):
            continue

    if best_result is None:
        return None

    A, omega, phi, C = best_result
    fitted    = sine_model(n, A, omega, phi, C)
    residuals = b - fitted

    return {
        'params'   : (A, omega, phi, C),
        'r_squared': best_r2,
        'fitted'   : fitted,
        'residuals': residuals,
    }


# ═══════════════════════════════════════════════════════════════
# 4. FIT ALL MU VALUES
# ═══════════════════════════════════════════════════════════════

fit_results = {}
for mu, b in results.items():
    fit_results[mu] = fit_sine_full(np.array(b))

mu_arr = np.array(sorted(fit_results.keys()))

# ── Extract phi and amplitude from the dict (consistent, one place) ──────
phi_arr = np.array([
    fit_results[mu]['params'][2] if fit_results[mu] is not None else np.nan
    for mu in mu_arr
])

amp_arr = np.array([
    fit_results[mu]['params'][0] if fit_results[mu] is not None else np.nan
    for mu in mu_arr
])

valid = ~np.isnan(phi_arr)

# ── Phase: subtract reference so first valid point = 0 ───────────────────
phi_plot = phi_arr.copy()
phi_plot[valid] -= phi_arr[valid][0]


# ═══════════════════════════════════════════════════════════════
# 5. COLOUR MAP SETUP
# ═══════════════════════════════════════════════════════════════

mu_c        = 2 * t          # critical point magnitude
crit_margin = 0.2 * t        # band half-width around ±2t

def _phase(mu):
    if abs(abs(mu) - mu_c) <= crit_margin:   # near +2t or -2t
        return 'critical'
    elif abs(mu) < mu_c:
        return 'topo'
    else:
        return 'trivial'

topo_mus    = [mu for mu in results.keys() if _phase(mu) == 'topo']
trivial_mus = [mu for mu in results.keys() if _phase(mu) == 'trivial']

topo_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo', ['#2d006b', '#6a00cc', '#aa44ff', '#dd99ff']
)
topo_norm = mcolors.Normalize(
    vmin=min(topo_mus) if topo_mus else -mu_c,
    vmax=max(topo_mus) if topo_mus else  mu_c,
)

trivial_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trivial', ['#fff176', '#ffd600', '#f9a825', '#6b4000']
)
trivial_norm = mcolors.Normalize(
    vmin= mu_c,
    vmax=max(trivial_mus) if trivial_mus else mu_c * 3,
)

def get_color(mu):
    phase = _phase(mu)
    if phase == 'critical':
        return 'black'
    elif phase == 'topo':
        return topo_cmap(topo_norm(mu))
    else:
        return trivial_cmap(trivial_norm(abs(mu)))  # symmetric for ±trivial

legend_entries = [
    mpatches.Patch(color='#6a00cc', label=r'Topological ($|\mu| < 2t$) — dark→light violet'),
    mpatches.Patch(color='black',   label=r'Critical region (near $\mu = \pm 2t$)'),
    mpatches.Patch(color='#ffd600', label=r'Trivial ($|\mu| > 2t$) — light→dark yellow'),
]

# ═══════════════════════════════════════════════════════════════
# 6. PLOT
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 6. PLOT
# ═══════════════════════════════════════════════════════════════

# ── Plot 1: Phase ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(mu_arr[valid], phi_plot[valid],
         '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for mu, phi in zip(mu_arr[valid], phi_plot[valid]):
    ax1.scatter(mu, phi, color=get_color(mu), s=20, zorder=3)

ax1.axvline( mu_c, color='red', ls='--', lw=1.2, label=r'$\mu = +2t$')
ax1.axvline(-mu_c, color='red', ls='--', lw=1.2, label=r'$\mu = -2t$')
ax1.axhline(0, color='gray', ls=':', lw=1.0)
ax1.axvspan(-mu_c, mu_c,        alpha=0.06, color='purple', label='Topological')
ax1.axvspan( mu_c, mu_arr.max(), alpha=0.06, color='yellow', label='Trivial')
ax1.axvspan(mu_arr.min(), -mu_c, alpha=0.06, color='yellow')

pi_ticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticks(pi_ticks)
ax1.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=10)
ax1.set_xlabel(r'$\mu$',              fontsize=13)
ax1.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=13)
ax1.set_title(rf'Phase shift $\Delta\phi$ vs $\mu$ — $N={N}$ sites', fontsize=13)
ax1.legend(handles=legend_entries, fontsize=8, loc='lower right', framealpha=0.88)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phase_vs_mu.png", dpi=150, bbox_inches='tight')
plt.show()

# ── Plot 2: Amplitude ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(mu_arr[valid], amp_arr[valid],
         '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for mu, amp in zip(mu_arr[valid], amp_arr[valid]):
    ax2.scatter(mu, amp, color=get_color(mu), s=20, zorder=3)

ax2.axvline( mu_c, color='red', ls='--', lw=1.2, label=r'$\mu = +2t$')
ax2.axvline(-mu_c, color='red', ls='--', lw=1.2, label=r'$\mu = -2t$')
ax2.axhline(0, color='gray', ls=':', lw=1.0)
ax2.axvspan(-mu_c, mu_c,        alpha=0.06, color='purple', label='Topological')
ax2.axvspan( mu_c, mu_arr.max(), alpha=0.06, color='yellow', label='Trivial')
ax2.axvspan(mu_arr.min(), -mu_c, alpha=0.06, color='yellow')

ax2.set_xlabel(r'$\mu$', fontsize=13)
ax2.set_ylabel(r'$A$',   fontsize=13)
ax2.set_title(rf'Fitted amplitude $A$ vs $\mu$ — $N={N}$ sites', fontsize=13)
ax2.legend(handles=legend_entries, fontsize=8, loc='lower right', framealpha=0.88)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("amplitude_vs_mu.png", dpi=150, bbox_inches='tight')
plt.show()
# %%
# ══════════════════════════════════════════════
# FIGURE 2 — all fitted curves
# ══════════════════════════════════════════════
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── Same colour helpers (reuse if already defined above) ──────
critical_band = (1.8, 3)

topo_mus    = [mu for mu in results.keys() if mu < critical_band[0]]
trivial_mus = [mu for mu in results.keys() if mu > critical_band[1]]

topo_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo', ['#6b0000', '#cc0000', '#ff4444', '#ff9999']
)
topo_norm = mcolors.Normalize(
    vmin=min(topo_mus) if topo_mus else 0,
    vmax=max(topo_mus) if topo_mus else 1
)

trivial_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trivial', ['#aaffaa', '#44cc44', '#1a8c1a', '#004d00']
)
trivial_norm = mcolors.Normalize(
    vmin=min(trivial_mus) if trivial_mus else 0,
    vmax=max(trivial_mus) if trivial_mus else 1
)

def get_color(mu):
    if critical_band[0] <= mu <= critical_band[1]:
        return 'black'
    elif mu < critical_band[0]:
        return topo_cmap(topo_norm(mu))
    else:
        return trivial_cmap(trivial_norm(mu))

# ── Plot ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 6))

for mu, b in results.items():
    color = get_color(mu)
    is_critical = critical_band[0] <= mu <= critical_band[1]
    lw    = 1.6 if is_critical else 1.0
    alpha_fit  = 0.95 if is_critical else 0.75
    alpha_raw  = 0.55 if is_critical else 0.35

    b    = np.array(b, dtype=float)
    n    = np.arange(len(b))
    popt = fit_results[mu]

    # raw data
    ax2.plot(n, b, 'o', color=color, ms=2, alpha=alpha_raw, zorder=1)

    # fitted curve
    if popt is not None:
        n_fine = np.linspace(0, len(b) - 1, 500)
        b_fit  = sine_model(n_fine, *popt)
        ax2.plot(n_fine, b_fit, '-', color=color,
                 lw=lw, alpha=alpha_fit, zorder=2)

# ── Custom legend ─────────────────────────────────────────────
legend_entries = [
    mpatches.Patch(color='#cc0000', label=r'Topological phase  ($\mu < 2t$)  — dark→light red'),
    mpatches.Patch(color='black',   label=r'Critical region  ($1.8 \leq \mu \leq 2.2$)'),
    mpatches.Patch(color='#1a8c1a', label=r'Trivial phase  ($\mu > 2t$)  — light→dark green'),
]
ax2.legend(handles=legend_entries, fontsize=10, loc='upper right', framealpha=0.88)

ax2.set_xlabel(r'$n$',   fontsize=13)
ax2.set_ylabel(r'$b_n$', fontsize=13)
ax2.set_title(r'Sine fits to $b_n$ for all $\mu$', fontsize=12)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# ══════════════════════════════════════════════
# FIGURE 3 — small multiples, one panel per mu
# ══════════════════════════════════════════════
n_mu  = len(mu_arr)
ncols = 5
nrows = int(np.ceil(n_mu / ncols))

fig3, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2.5))
axes = axes.flatten()

for i, (mu, color) in enumerate(zip(mu_arr, colors)):
    ax  = axes[i]
    b   = np.array(results[mu], dtype=float)
    n   = np.arange(len(b))
    popt = fit_results[mu]

    ax.plot(n, b, 'o', color=color, ms=2, alpha=0.5)
    if popt is not None:
        n_fine = np.linspace(0, len(b)-1, 500)
        ax.plot(n_fine, sine_model(n_fine, *popt), '-', color=color, lw=1.5)
        ax.set_title(rf'$\mu={mu:.2f}$, $\phi={popt[2]:.2f}$', fontsize=8)
    else:
        ax.set_title(rf'$\mu={mu:.2f}$ — fit failed', fontsize=8, color='red')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

# hide unused panels
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig3.suptitle(r'Sine fits to $b_n$ — one panel per $\mu$ for $N$ lattice Sites', fontsize=13)
plt.tight_layout()
plt.show()


# %%
# Get phi values just before and just after mu = 2
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.cm as cm

# ── (keep your existing sine_model, fit_sine_full, fit_results as-is) ──

# ══════════════════════════════════════════════════════════════════
# QUANTIZATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def compute_phase_jump(mu_arr, phi_norm, transition=2.0, window=0.5):
    """
    Estimate the phase jump at the transition by averaging phi 
    in a window on each side, away from the critical point.
    
    window: how far from transition to start averaging (in mu units)
    Returns: jump value, mean_left, mean_right, std_left, std_right
    """
    # Avoid the critical region (transition ± window) to get clean averages
    left_mask  = (mu_arr < transition - window) & ~np.isnan(phi_norm)
    right_mask = (mu_arr > transition + window) & ~np.isnan(phi_norm)

    mean_left  = np.mean(phi_norm[left_mask])
    mean_right = np.mean(phi_norm[right_mask])
    std_left   = np.std(phi_norm[left_mask])
    std_right  = np.std(phi_norm[right_mask])

    jump = mean_right - mean_left
    return jump, mean_left, mean_right, std_left, std_right


jump, mean_L, mean_R, std_L, std_R = compute_phase_jump(mu_arr, phi_norm)

print(f"Phase in topological phase  (μ < 2t): {mean_L:.4f} ± {std_L:.4f} rad")
print(f"Phase in trivial phase      (μ > 2t): {mean_R:.4f} ± {std_R:.4f} rad")
print(f"Phase jump Δφ                        : {jump:.4f} rad")
print(f"Jump / π                             : {jump/np.pi:.4f}")
print(f"Deviation from π                     : {abs(abs(jump) - np.pi):.4f} rad  ({abs(abs(jump)/np.pi - 1)*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════
# FIGURE — 3-panel quantization summary
# ══════════════════════════════════════════════════════════════════

fig, ax2 = plt.subplots(figsize=(7, 5))

# Raw data
ax2.plot(mu_arr[valid], phi_norm[valid], 'o', color='steelblue', ms=3.5, alpha=0.5, label='Raw')

# Plateau bands
ax2.axhspan(mean_L - std_L, mean_L + std_L, alpha=0.25, color='blue')
ax2.axhspan(mean_R - std_R, mean_R + std_R, alpha=0.25, color='orange')
ax2.axhline(mean_L, color='blue',   lw=1.5, ls='-',  label=fr'Left plateau = {mean_L:.3f}')
ax2.axhline(mean_R, color='orange', lw=1.5, ls='-',  label=fr'Right plateau = {mean_R:.3f}')
ax2.axvline(2.0,    color='red',    lw=1.2, ls='--', label=r'$\mu = 2t$')

# Jump annotation
ax2.annotate('', xy=(2.6, mean_R), xytext=(2.6, mean_L),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(2.65, (mean_L + mean_R) / 2,
         fr'$|\Delta\phi|$ = {abs(jump):.3f}' + '\n' + fr'= {abs(jump)/np.pi:.3f}$\pi$',
         fontsize=9, va='center')

# Phase shading
ax2.axvspan(mu_arr.min(), 2.0, alpha=0.07, color='blue',   label='Topological')
ax2.axvspan(2.0, mu_arr.max(), alpha=0.07, color='orange', label='Trivial')

pi_ticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticks(pi_ticks)
ax2.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=10)

ax2.set_xlabel(r'$\mu$', fontsize=13)
ax2.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=13)
ax2.set_title(
    rf'Phase jump quantization — Kitaev chain, $N={N}$ sites',
    fontsize=12
)
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# Annotate deviation



# %%

# %%