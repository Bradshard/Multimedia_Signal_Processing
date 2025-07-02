import numpy as np
import json
import os

# cost func
def cost(a1, a2):
    a0 = 1 - a1 - a2
    G = a0 + a1*np.cos(thetas) + a2*np.cos(2*thetas)
    G_sq = G**2
    in_m = thetas <= alpha
    out_m = thetas > alpha
    
    leak = np.trapezoid(G_sq[out_m], thetas[out_m])/(np.pi-alpha)
    unif = np.trapezoid((G_sq[in_m]-1)**2, thetas[in_m])/alpha
    return lam*leak + (1-lam)*unif

# Cost function for cardioid and grid-search params
alpha  = np.pi/4
lam = 1
M = 360
thetas = np.linspace(0, np.pi, M)

# grid-search for best vals
grid = np.linspace(0,1,201) # 200 grid is good enough
eps = 1e-8 # to prevent error for a2 div error.
best = (np.inf, 0.0,0.0)
for a1 in grid:
    for a2 in grid:
        if a1 + a2 <= 1 and a2 >= eps:
            val = cost(a1,a2)
            if val < best[0]:
                best = (val, a1, a2)
_, a1_opt, a2_opt = best # DOF is 2 due to sum is 1 so first one is depending on a1 and a2, so no use.
a0_opt = 1 - a1_opt - a2_opt

#### According to Eqs.(30–32) in paper derivation of delays & weights
d, c, fs = 0.0254, 343.0, 48000
tau = (a1_opt) / (2*a2_opt) * (d/c) # seconds
D = tau * fs # samples
wL = wR = a2_opt
wC = a0_opt - (a1_opt**2)/(4*a2_opt)
tau_ms = tau *1e3 # ms

# For reading and JSON
os.makedirs("beamformed", exist_ok=True)
txt = """Optimal 2nd-order DMA coefficients & delay
α = {alpha:.3f} rad, λ = {lam:.3f}

a0 = {a0:.4f}
a1 = {a1:.4f}
a2 = {a2:.4f}

Delay D = {D:.2f} samples (τ = {tau_ms:.2f} ms)
wL = wR = {wL:.4f}
wC = {wC:.4f}
""".format(alpha=alpha, lam=lam, a0=a0_opt, a1=a1_opt, a2=a2_opt,
           D=D, tau_ms=tau_ms, wL=wL, wC=wC)

with open("beamformed/coeffs_for_weights_delay.txt","w") as f:
    f.write(txt)

with open("beamformed/params.json","w") as f:
    json.dump({
        "a0":a0_opt, "a1":a1_opt, "a2":a2_opt,
        "D":D, "wL":wL, "wC":wC, "wR":wR
    }, f, indent=2)

print("Design saved to coeffs_for_weights_delay.txt and params.json")
