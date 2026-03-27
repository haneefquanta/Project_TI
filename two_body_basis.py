# %%
import numpy as np
import matplotlib.pyplot as plt
from Kitaev_realspace import build_HM_blocks
from Kitaev_realspace import Hamiltonian
import matplotlib.cm as cm

mu_values = [1,2,3]



N = 20
BC = 0
O_init = np.zeros((2*N,2*N), dtype=complex)
site = 0    # mi




#O_init[0,N] = -1
#O_init[N,0] = 1

O_init[0,2*N - 1] = -1
O_init[2*N -1 ,0] = 1


def inner(A, B):
    return np.trace(A.conj().T @ B)




def lanczos(H, O_init,window=2*N*(N-1)):
    dim    = H.shape[0]
    krylov_basis = []
    a_coeffs = []
    b_coeffs = [0.0]

    

    O_n = O_init / np.linalg.norm(O_init)
    krylov_basis.append(O_n.copy())
    O_prev = np.zeros((dim, dim))

    for n in range(2*N*(N-1)):
        
        

        A = (H @ O_n - O_n @ H) * 1j
        
        
        a_n = np.real(inner(O_n,A))          # ← fixed inner product
        a_coeffs.append(a_n)

        A = A - a_n * O_n - b_coeffs[n] * O_prev
        
        #   reorthogonalization 
        for O_k in krylov_basis[-window:]: 
            A = A -inner(O_k,A) * O_k
        for O_k in krylov_basis[-window:]:   
            A = A -inner(O_k,A) * O_k
    

        b_next = np.linalg.norm(A)

        if b_next < 1e-7:
            print(f"Krylov space exhausted at K = {n+1}")
            break

        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n    = A / b_next
        krylov_basis.append(O_n.copy())

        # ── keep only last `window` vectors ──────────────
        if len(krylov_basis) > window:
            krylov_basis.pop(0)

        # record sector content of current basis vector
       

    b_coeffs = b_coeffs[1:]
    return np.array(b_coeffs) 


results = {}
for mu in mu_values:
    print(f"Running μ = {mu} ...", end="  ")
    H_m = build_HM_blocks(N, mu, 1, 1, BC,-1)
    b   = lanczos(H_m, O_init) 
    results[mu] = b                         
    print(f"done  (K = {len(b)})")
colors = cm.plasma(np.linspace(0.1, 0.9, len(mu_values)))

from scipy.ndimage import uniform_filter1d


fig, ax = plt.subplots(figsize=(10, 5))

for (mu, b), color in zip(results.items(), colors):
    label = rf'$\mu = {mu}$' + (' ← critical' if mu == 2.0 else '')
    n_b   = np.arange(1, len(b) + 1)

    b_smooth = uniform_filter1d(b, size=10)

    
    ax.scatter(n_b, b, color=color, s=4, alpha=0.2, zorder=1)


    ax.plot(n_b, b_smooth, '-', color=color, linewidth=2,
            label=label, alpha=0.9, zorder=2)

ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel(r'$n$',   fontsize=13)
ax.set_ylabel(r'$b_n$', fontsize=13)
ax.set_title(rf'Lanczos coefficients — $N={N}$, BC: {BC}', fontsize=12)
ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# %%
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm


def lanczos_store(H, O_init, window=2*N*(N-1)):
    dim = H.shape[0]
    krylov_vectors = []
    b_coeffs = [0.0]
    O_n = O_init / np.linalg.norm(O_init)
    krylov_vectors.append(O_n.copy())
    O_prev = np.zeros((dim, dim))
    for n in range(2*N*(N-1)):
        A = (H @ O_n - O_n @ H) * 1j
        a_n = np.real(inner(O_n, A))
        A = A - a_n * O_n - b_coeffs[n] * O_prev
        for O_k in krylov_vectors[-window:]:
            A = A - inner(O_k, A) * O_k
        for O_k in krylov_vectors[-window:]:
            A = A - inner(O_k, A) * O_k
        b_next = np.linalg.norm(A)
        if b_next < 1e-7:
            print(f"μ exhausted at K={n+1}")
            break
        b_coeffs.append(b_next)
        O_prev = O_n.copy()
        O_n = A / b_next
        krylov_vectors.append(O_n.copy())
        if len(krylov_vectors) > window:
            krylov_vectors.pop(0)
    return krylov_vectors

krylov = {}
for mu in mu_values:
    print(f"μ={mu} ...", end="  ")
    H_m = build_HM_blocks(N, mu, 1, 1, BC, -1)
    krylov[mu] = lanczos_store(H_m, O_init)
    print(f"done (K={len(krylov[mu])})")

# ── lattice positions ─────────────────────────────────────────────────────────
A_pos = np.array([[i, 1.0] for i in range(N)])
B_pos = np.array([[i, 0.0] for i in range(N)])

def idx_to_pos(idx):
    return A_pos[idx] if idx < N else B_pos[idx - N]

K_max = max(len(krylov[mu]) for mu in mu_values)

# ── setup figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(mu_values), figsize=(6*len(mu_values), 5))

# pre-create one line per bond (i,j) with i<j
bonds = [(i, j) for i in range(2*N) for j in range(i+1, 2*N)]
line_collections = {}

for ax, mu in zip(axes, mu_values):
    lines = [ax.plot([], [], color='k', lw=0)[0] for _ in bonds]
    ax.scatter(A_pos[:, 0], A_pos[:, 1], s=80, color='royalblue', zorder=5)
    ax.scatter(B_pos[:, 0], B_pos[:, 1], s=80, color='tomato',    zorder=5)
    for i in range(N):
        ax.text(i,  1.12, f"{i+1}A", ha='center', fontsize=6, color='royalblue')
        ax.text(i, -0.15, f"{i+1}B", ha='center', fontsize=6, color='tomato')
    ax.set_xlim(-1, N); ax.set_ylim(-0.4, 1.4); ax.axis('off')
    line_collections[mu] = lines

titles = {mu: ax.set_title("") for mu, ax in zip(mu_values, axes)}

def update(n):
    for mu, ax in zip(mu_values, axes):
        kvecs = krylov[mu]
        O_n = kvecs[n] if n < len(kvecs) else np.zeros((2*N, 2*N))
        
        # weight matrix: symmetrized absolute value
        W = np.abs(O_n) + np.abs(O_n.T)
        W_max = W.max() if W.max() > 0 else 1.0
        W = W / W_max

        for k, (i, j) in enumerate(bonds):
            w = W[i, j]
            line = line_collections[mu][k]
            if w < 1e-2:
                line.set_data([], [])
            else:
                p1 = idx_to_pos(i)
                p2 = idx_to_pos(j)
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_linewidth(w * 4)
                line.set_alpha(float(min(w * 2, 1.0)))
                line.set_color(cm.plasma(float(w)))

        titles[mu].set_text(f"μ={mu}  |  n={n}")
    return [l for lines in line_collections.values() for l in lines]

anim = FuncAnimation(fig, update, frames=K_max, interval=80, blit=True)
plt.suptitle("Krylov operator flow on Majorana lattice", fontsize=13)
plt.tight_layout()
anim.save("krylov_flow.gif", writer='pillow', fps=15, dpi=70)
plt.show()

# %%

# %%