# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from numpy import unwrap
mu_values = np.linspace(-7, 7, 100)

t1    = 1
t2    = 1
delta = 1
N     = 30
BC    = 0
w     = 30
D     = 0
epsilon = 0

# ── Phase boundaries (gap closes at k=0 and k=π) ─────────────────────────────
# k=0  → μ = ±2(t1+t2)
# k=π  → μ = ±2(t1−t2)   [= 0 when t1=t2]
mu_c_outer = 2 * (t1 + t2)          # = 4
mu_c_inner = 2 * abs(t1 - t2)       # = 0  (when t1==t2)

# Half-width of the "critical band" drawn around each transition point
crit_hw = 0.25

# ── Seed operators ────────────────────────────────────────────────────────────
O_bulk = np.zeros((2*N, 2*N), dtype=complex)
mid = N // 2 + 1
O_bulk[mid, mid + N] = -1
O_bulk[mid + N, mid] =  1



O_first = np.zeros((2*N,2*N), dtype=complex)
O_first[0,N+2] = 1
O_first[N+2,0] = -1

O_edge = np.zeros((2*N,2*N), dtype=complex)
O_edge[0,2*N - 1] = -1
O_edge[2*N -1 ,0] = 1

O_init = O_bulk
O_init = O_edge
#O_init =O_first

# ── Lanczos ───────────────────────────────────────────────────────────────────
def lanczos(H, O_init, window=w):
    dim = H.shape[0]
    krylov_basis = []
    a_coeffs  = []
    b_coeffs  = [0.0]

    O_n   = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros((dim, dim))

    for n in range(window):
        A   = (H @ O_n - O_n @ H) * 1j
        a_n = np.real(np.sum(O_n.conj() * A))
        a_coeffs.append(a_n)
        A   = A - a_n * O_n - b_coeffs[n] * O_prev

        for O_k in krylov_basis[-window:]:
            A = A - np.sum(O_k.conj() * A) * O_k

        b_next = np.linalg.norm(A)
        if b_next < 1e-3:
            print(f"  Krylov exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

    return np.array(b_coeffs[1:]), krylov_basis


# ── NNN Kitaev Hamiltonian ────────────────────────────────────────────────────
def NNN_kitaev(N, mu, t1, t2, delta, BC, parity, epsilon, W):
    mu_i = mu + W * np.random.uniform(-1, 1, N)
    H_AB = np.zeros((N, N), dtype=complex)

    for j in range(N):
        H_AB[j, j] = -mu_i[j] / 2

    for j in range(N - 1):
        H_AB[j,   j+1] =  (delta - t1) / 2 + 1j * epsilon
        H_AB[j+1, j  ] = -(t1 + delta) / 2 - 1j * epsilon

    for j in range(N - 2):
        H_AB[j,   j+2] = -t2 / 2
        H_AB[j+2, j  ] = -t2 / 2

    if BC == 1:
        H_AB[N-1, 0] = -parity * ( (delta - t1) / 2) + 1j * epsilon
        H_AB[0, N-1] = -parity * (-(t1 + delta) / 2) - 1j * epsilon
        for (a, b) in [(0, N-2), (N-2, 0), (1, N-1), (N-1, 1)]:
            H_AB[a, b] = -parity * (-t2 / 2)

    H_m = np.block([[np.zeros((N, N)),  H_AB        ],
                    [-H_AB.T,           np.zeros((N, N))]])
    return H_m


# ── Run Lanczos for all μ values ──────────────────────────────────────────────
results    = {}
results_kb = {}

def get_asymptotic_scale(b, tail=5):
    """Use mean of last `tail` b_n values as normalization scale."""
    return np.mean(np.abs(b[-tail:]))


for mu in mu_values:
    print(f"μ = {mu:+.3f}", end="  ")
    H_m = NNN_kitaev(N, mu, t1, t2, delta, BC, 1, epsilon, D)
    b, K_b = lanczos(H_m, O_init)
    # Normalise by gap at k=0: |μ/2 + t1 + t2|  (+ δ² term for safety)
    scale = np.sqrt(((mu / 2) + t1 + t2)**2 + delta**2)
    scale = get_asymptotic_scale(b, tail=5)
    results[mu]    = b / scale
    results_kb[mu] = K_b
    print(f"done (K = {len(b)})")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: phase label for a given μ
# ═══════════════════════════════════════════════════════════════════════════════
def _phase(mu):
    """
    NNN Kitaev phase diagram (t1, t2 > 0, gap closes at k=0 and k=π):
      trivial  : |μ| > 2(t1+t2)
      topo_w1  : 2|t1−t2| < |μ| < 2(t1+t2)     [winding w=1]
      topo_w2  : |μ| < 2|t1−t2|                 [winding w=2]
                 (collapses to a point when t1==t2)
    Critical bands are placed around each transition.
    """
    abs_mu = abs(mu)
    # near outer boundary ±2(t1+t2)
    if abs(abs_mu - mu_c_outer) <= crit_hw:
        return 'crit_outer'
    # near inner boundary ±2|t1−t2|  (only meaningful when t1 ≠ t2)
    if mu_c_inner > 0 and abs(abs_mu - mu_c_inner) <= crit_hw:
        return 'crit_inner'
    # near μ=0 transition (always present; when t1==t2 this IS the inner boundary)
    if abs(mu) <= crit_hw:
        return 'crit_inner'
    # bulk phases
    if abs_mu > mu_c_outer:
        return 'trivial'
    if mu_c_inner > 0 and abs_mu < mu_c_inner:
        return 'topo_w2'
    return 'topo_w1'


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Lanczos b_n coefficients
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Colour maps per phase
topo_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo_w1', ['#6b0000', '#cc0000', '#ff4444', '#ff9999'])
topo2_cmap = mcolors.LinearSegmentedColormap.from_list(
    'topo_w2', ['#00008b', '#0000ff', '#6699ff', '#bbccff'])
trivial_cmap = mcolors.LinearSegmentedColormap.from_list(
    'trivial', ['#aaffaa', '#44cc44', '#1a8c1a', '#004d00'])

topo_mus    = [mu for mu in results if _phase(mu) == 'topo_w1']
topo2_mus   = [mu for mu in results if _phase(mu) == 'topo_w2']
trivial_mus = [mu for mu in results if _phase(mu) == 'trivial']

topo_norm    = mcolors.Normalize(vmin=min(topo_mus)    if topo_mus    else 0,
                                  vmax=max(topo_mus)    if topo_mus    else 1)
topo2_norm   = mcolors.Normalize(vmin=min(topo2_mus)   if topo2_mus   else 0,
                                  vmax=max(topo2_mus)   if topo2_mus   else 1)
trivial_norm = mcolors.Normalize(vmin=min(trivial_mus) if trivial_mus else 0,
                                  vmax=max(trivial_mus) if trivial_mus else 1)

def get_color_bn(mu):
    phase = _phase(mu)
    if 'crit' in phase:
        return 'black'
    if phase == 'topo_w1':
        return topo_cmap(topo_norm(mu))
    if phase == 'topo_w2':
        return topo2_cmap(topo2_norm(mu))
    return trivial_cmap(trivial_norm(abs(mu)))

fig, ax = plt.subplots(figsize=(11, 5))

for mu, b in results.items():
    color = get_color_bn(mu)
    is_crit = 'crit' in _phase(mu)
    lw    = 1.6 if is_crit else 0.7
    alpha = 0.95 if is_crit else 0.65
    n_b   = np.arange(1, len(b) + 1)
    ax.scatter(n_b, b, color=color, s=3,  alpha=alpha, zorder=2)
    ax.plot(   n_b, b, '-', color=color, linewidth=lw, alpha=alpha, zorder=1)

ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)

legend_entries = [
    mpatches.Patch(color='#cc0000', label=r'Topological $w{=}1$  ($2|t_1{-}t_2| < |\mu| < 2(t_1{+}t_2)$)'),
    mpatches.Patch(color='#0000ff', label=r'Topological $w{=}2$  ($|\mu| < 2|t_1{-}t_2|$)  [collapses when $t_1{=}t_2$]'),
    mpatches.Patch(color='black',   label=r'Critical  ($\mu \approx 0,\ \pm 2(t_1{+}t_2)$)'),
    mpatches.Patch(color='#1a8c1a', label=r'Trivial  ($|\mu| > 2(t_1{+}t_2)$)'),
]
ax.legend(handles=legend_entries, fontsize=9, loc='upper right', framealpha=0.88)
ax.set_xlabel(r'$n$', fontsize=13)
ax.set_ylabel(r'normalised $b_n$', fontsize=13)
ax.set_title(
    rf'Lanczos coefficients — NNN Kitaev, $N={N}$, '
    rf'$t_1={t1}$, $t_2={t2}$, $\Delta={delta}$, BC: {BC}',
    fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lanczos_bn.png", dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PHI / AMPLITUDE FITTING
# ═══════════════════════════════════════════════════════════════════════════════

# drop first 3 transient steps
results_fit = {mu: b[3:] for mu, b in results.items()}

from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import sigmaclip


def sine_model(n, A, omega, phi, C):
    return A * np.sin(omega * n + phi) + C


def _top_k_fft_freqs(b_c, k=5):
    freqs = fftfreq(len(b_c), d=1)
    power = np.abs(fft(b_c))
    pos   = freqs > 0
    top   = np.argsort(power[pos])[::-1][:k]
    return 2 * np.pi * freqs[pos][top]


def _refine_omega(b, omega_c, frac=0.3):
    n  = np.arange(len(b))
    lo = max(omega_c * (1 - frac), 1e-6)
    hi = omega_c * (1 + frac)
    def res(omega):
        X = np.column_stack([np.sin(omega*n), np.cos(omega*n), np.ones(len(n))])
        beta, *_ = np.linalg.lstsq(X, b, rcond=None)
        return np.sum((b - X @ beta)**2)
    return minimize_scalar(res, bounds=(lo, hi), method='bounded',
                           options={'xatol': 1e-8}).x


def _analytical_init(b, omega):
    n  = np.arange(len(b))
    X  = np.column_stack([np.sin(omega*n), np.cos(omega*n), np.ones(len(n))])
    a, bc, C, *_ = (*np.linalg.lstsq(X, b, rcond=None)[0],)
    A   = np.sqrt(a**2 + bc**2)
    phi = np.arctan2(bc, a)
    return A, phi, C


def _r2(b, b_pred):
    ss_res = np.sum((b - b_pred)**2)
    ss_tot = np.sum((b - b.mean())**2)
    return 1.0 - ss_res / ss_tot if ss_tot else (1.0 if ss_res == 0 else 0.0)


def fit_sine_full(b, k_candidates=5):
    b = np.array(b, dtype=float)
    n = np.arange(len(b))
    if len(b) < 4:
        raise ValueError("Need ≥4 points.")

    b_c        = b - b.mean()
    amp_range  = b.max() - b.min()
    candidates = []
    for omega_c in _top_k_fft_freqs(b_c, k=k_candidates):
        try:
            candidates.append(_refine_omega(b, omega_c))
        except Exception:
            candidates.append(omega_c)

    best, best_r2 = None, -np.inf
    for omega_g in candidates:
        try:
            A_g, phi_g, C_g = _analytical_init(b, omega_g)
            p0 = [A_g, omega_g, phi_g, C_g]
            bounds = ([0., omega_g*0.5, -2*np.pi, b.min()],
                      [amp_range+1e-6, omega_g*1.5, 2*np.pi, b.max()])
            popt, _ = curve_fit(sine_model, n, b, p0=p0, bounds=bounds,
                                maxfev=200000, ftol=1e-12, xtol=1e-12)
            r2 = _r2(b, sine_model(n, *popt))
            if r2 > best_r2:
                best_r2, best = r2, popt
        except (RuntimeError, ValueError):
            continue

    if best is None:
        return None
    A, omega, phi, C = best
    fitted = sine_model(n, A, omega, phi, C)
    return {'params': (A, omega, phi, C), 'r_squared': best_r2,
            'fitted': fitted, 'residuals': b - fitted}


fit_results = {mu: fit_sine_full(np.array(b)) for mu, b in results_fit.items()}

mu_arr  = np.array(sorted(fit_results.keys()))
phi_arr = np.array([fit_results[mu]['params'][2]
                    if fit_results[mu] else np.nan for mu in mu_arr])
amp_arr = np.array([fit_results[mu]['params'][0]
                    if fit_results[mu] else np.nan for mu in mu_arr])
valid   = ~np.isnan(phi_arr)

phi_plot          = phi_arr.copy()
phi_plot[valid]  -= phi_arr[valid][0]




# ── Colour map for phi/amplitude plots ───────────────────────────────────────
topo_cmap2 = mcolors.LinearSegmentedColormap.from_list(
    'topo2', ['#2d006b', '#6a00cc', '#aa44ff', '#dd99ff'])
topo2_cmap2 = mcolors.LinearSegmentedColormap.from_list(
    'topo2b', ['#00008b', '#3355ff', '#88aaff', '#ccddff'])
trivial_cmap2 = mcolors.LinearSegmentedColormap.from_list(
    'triv2', ['#fff176', '#ffd600', '#f9a825', '#6b4000'])

topo_norm2   = mcolors.Normalize(vmin=-mu_c_outer, vmax=mu_c_outer)
trivial_norm2 = mcolors.Normalize(vmin=mu_c_outer, vmax=mu_arr.max())

def get_color_phi(mu):
    phase = _phase(mu)
    if 'crit' in phase:
        return 'black'
    if phase == 'topo_w2':
        return topo2_cmap2(topo2_norm(mu))
    if phase == 'topo_w1':
        return topo_cmap2(topo_norm2(mu))
    return trivial_cmap2(trivial_norm2(abs(mu)))

legend_phi = [
    mpatches.Patch(color='#6a00cc', label=r'Topo $w{=}1$  ($2|t_1{-}t_2|<|\mu|<2(t_1{+}t_2)$)'),
    mpatches.Patch(color='#3355ff', label=r'Topo $w{=}2$  ($|\mu|<2|t_1{-}t_2|$)  [point when $t_1{=}t_2$]'),
    mpatches.Patch(color='black',   label=r'Critical'),
    mpatches.Patch(color='#ffd600', label=r'Trivial  ($|\mu|>2(t_1{+}t_2)$)'),
]


def _draw_phase_regions(ax, mu_arr):
    """Add vertical lines and shaded regions for all phase boundaries."""
    mu_min, mu_max = mu_arr.min(), mu_arr.max()

    # Shade trivial region
    for x_lo, x_hi in [(-mu_c_outer, mu_min), (mu_c_outer, mu_max)]:
        if x_lo < x_hi:
            ax.axvspan(x_lo, x_hi, alpha=0.07, color='gold')

    # Shade topological w=1 region (between inner and outer boundaries)
    lo1, hi1 = -mu_c_outer, -mu_c_inner if mu_c_inner > 0 else 0
    lo2, hi2 =  mu_c_inner if mu_c_inner > 0 else 0,  mu_c_outer
    ax.axvspan(lo1, lo2, alpha=0.07, color='purple')
    ax.axvspan(hi1, hi2, alpha=0.07, color='purple')

    # Shade topological w=2 region (inside inner boundaries)
    if mu_c_inner > 0:
        ax.axvspan(-mu_c_inner, mu_c_inner, alpha=0.10, color='blue')

    # Vertical lines at every phase boundary
    boundaries = {
        mu_c_outer:   (r'$\mu{=}+2(t_1{+}t_2)$', 'right'),
        -mu_c_outer:  (r'$\mu{=}-2(t_1{+}t_2)$', 'left'),
    }
    if mu_c_inner > 0:
        boundaries[ mu_c_inner] = (r'$\mu{=}+2|t_1{-}t_2|$', 'right')
        boundaries[-mu_c_inner] = (r'$\mu{=}-2|t_1{-}t_2|$', 'left')
    else:
        # t1==t2: single transition at μ=0
        boundaries[0.0] = (r'$\mu{=}0$', 'right')

    for xv, (lbl, side) in boundaries.items():
        ax.axvline(xv, color='red', ls='--', lw=1.2)
        ylims = ax.get_ylim()
        ypos  = ylims[0] + 0.05 * (ylims[1] - ylims[0])
        ha    = side
        ax.text(xv + (0.05 if side == 'right' else -0.05), ypos,
                lbl, color='red', fontsize=7, ha=ha, va='bottom',
                rotation=90, alpha=0.8)


# ── Plot 2: Phase φ vs μ ──────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(mu_arr[valid], phi_plot[valid], '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for mu, phi in zip(mu_arr[valid], phi_plot[valid]):
    ax1.scatter(mu, phi, color=get_color_phi(mu), s=20, zorder=3)

_draw_phase_regions(ax1, mu_arr)

pi_ticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticks(pi_ticks)
ax1.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=10)
ax1.set_xlabel(r'$\mu$', fontsize=13)
ax1.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=13)
ax1.set_title(
    rf'Phase shift $\Delta\phi$ vs $\mu$ — NNN Kitaev, $N={N}$, '
    rf'$t_1={t1}$, $t_2={t2}$, $\Delta={delta}$',
    fontsize=12)
ax1.legend(handles=legend_phi, fontsize=8, loc='upper right', framealpha=0.88)
ax1.axhline(0, color='gray', ls=':', lw=1.0)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phase_vs_mu.png", dpi=150, bbox_inches='tight')
plt.show()


# ── Plot 3: Amplitude A vs μ ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.plot(mu_arr[valid], amp_arr[valid], '-', color='gray', lw=0.8, alpha=0.5, zorder=1)
for mu, amp in zip(mu_arr[valid], amp_arr[valid]):
    ax2.scatter(mu, amp, color=get_color_phi(mu), s=20, zorder=3)

_draw_phase_regions(ax2, mu_arr)

ax2.set_xlabel(r'$\mu$', fontsize=13)
ax2.set_ylabel(r'$A$', fontsize=13)
ax2.set_title(
    rf'Fitted amplitude $A$ vs $\mu$ — NNN Kitaev, $N={N}$, '
    rf'$t_1={t1}$, $t_2={t2}$, $\Delta={delta}$',
    fontsize=12)
ax2.legend(handles=legend_phi, fontsize=8, loc='upper right', framealpha=0.88)
ax2.axhline(0, color='gray', ls=':', lw=1.0)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("amplitude_vs_mu.png", dpi=150, bbox_inches='tight')
plt.show()
# %%
