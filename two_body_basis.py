# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit          # ← add here at the top



mu_values =np.linspace(0.2,6,10)
#mu_values = [10,0.5]
t= 1
delta = 1

N = 50
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
O_init =+ O_edge




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
        for O_k in krylov_basis[-window:]:   
            A = A - np.sum(O_k.conj() * A) * O_k
        
        
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

critical_band = (1.8, 2.2)

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
#plt.show()







# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.cm as cm

def sine_model(n, A, omega, phi, C):
    return A * np.sin(omega * n + phi) + C

for mu in mu_values:
    results[mu] = results[mu][3:]


def fit_sine_full(b):
    b = np.array(b, dtype=float)
    n = np.arange(len(b))
    b_centered = b - b.mean()
    freqs      = fftfreq(len(b), d=1)
    power      = np.abs(fft(b_centered))
    pos_mask   = freqs > 0
    dominant_freq = freqs[pos_mask][np.argmax(power[pos_mask])]
    omega_guess   = 2 * np.pi * dominant_freq

    A_guess = (b.max() - b.min()) / 2
    C_guess = b.mean()

    p0     = [A_guess, omega_guess, 0.0, C_guess]
    bounds = (
        [0.0,            omega_guess * 0.5, -np.pi, b.min()],
        [b.max()-b.min(), omega_guess * 1.5,  np.pi, b.max()]
    )
    try:
        popt, _ = curve_fit(sine_model, n, b, p0=p0, bounds=bounds, maxfev=100000)
        return popt   # A, omega, phi, C
    except (RuntimeError, ValueError):
        return None

# ── Fit all mu ──
fit_results = {}
for mu, b in results.items():
    popt = fit_sine_full(np.array(b))
    fit_results[mu] = popt

mu_arr  = np.array(sorted(fit_results.keys()))
phi_arr = np.array([fit_results[mu][2] if fit_results[mu] is not None else np.nan for mu in mu_arr])
valid   = ~np.isnan(phi_arr)

colors = cm.plasma(np.linspace(0.1, 0.9, len(mu_arr)))

# ══════════════════════════════════════════════
# FIGURE 1 — phi vs mu
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# FIGURE 1 — phi vs mu  (first mu as reference φ = 0)
# ══════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(9, 5))

# Normalize: shift so the first valid phi = 0
phi_ref  = phi_arr[valid][0]          # reference value (first valid mu)
phi_norm = phi_arr.copy()
phi_norm[valid] -= phi_ref            # subtract reference from all valid points

ax1.plot(mu_arr[valid], phi_norm[valid], 'o-', color='steelblue', ms=4, lw=1.2)
ax1.axvline(2,  color='red',  ls='--', lw=1, label=r'$\mu = 2t$')
ax1.axhline(0,  color='gray', ls=':',  lw=1)
ax1.set_xlabel(r'$\mu$',                           fontsize=13)
ax1.set_ylabel(r'$\Delta\phi$ (rad)',              fontsize=13)
ax1.set_title(
    rf'Phase shift $\Delta\phi$ of $b_n$ vs $\mu$ '
    rf'(reference: first $\mu$)  {N} lattice Sites',
    fontsize=12
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ══════════════════════════════════════════════
# FIGURE 2 — all fitted curves
# ══════════════════════════════════════════════
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── Same colour helpers (reuse if already defined above) ──────
critical_band = (1.8, 2.2)

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
mu_before = mu_arr[valid][mu_arr[valid] < 1.7][-1]
mu_after  = mu_arr[valid][mu_arr[valid] > 2.3][0]

phi_before = phi_norm[valid][mu_arr[valid] < 1.7][-1]
phi_after  = phi_norm[valid][mu_arr[valid] > 2.3][0]

print(f"Δφ at transition = {phi_after - phi_before:.4f} rad")
print(f"Δφ / π           = {(phi_after - phi_before) / np.pi:.4f}")
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