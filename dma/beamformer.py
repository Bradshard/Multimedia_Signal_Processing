import numpy as np
from scipy.signal import lfilter

# Beamformer design as specified : delay + filter + sum

# Fractional-delay function using linear interpolation
def frac_delay(x, D):
    """
    Fractional delay of D samples to signal x by linear interpolation.

    Parameters:
    x (array): Input signal array.
    D (float): Delay amount in samples (can be fractional).

    Returns:
    array: Delayed signal of same length as x.
    """
    # sample indices
    n = np.arange(len(x))
    # interpolate x at n - D, zero-padding(fill as in cnn) outside of bounds
    return np.interp(n, n - D, x, left=0.0, right=0.0)
"""
# Al-Alaoui first‐order integrator
#    H(z) = ( (1–a) + a*zˆ-1 ) / (1 – zˆ-1)
#    choose a=0.5 for a balanced system
# —————————————————————————————————————————————————————————
def alaoui_integrator(x, alpha=0.5):
    # this used because it was used in the paper.
    b = [1 - alpha, alpha]   # numerator [1–a, a]
    a = [1.0, -1.0]          # denominator 1 – z**-1
    return lfilter(b, a, x)
    

# Simpson integrator (second‐order accurate trapezoid‐like)
#    H(z) = (1/6)*(1 + 4 zˆ-1 + zˆ-2)/(1 – zˆ-1)
def simpson_integrator(x):
    # same as al-alaoui
    b = [1/6, 4/6, 1/6]  # FIR taps [1,4,1]/6
    a = [1.0, -1.0]      # denominator 1 – zˆ-1
    return lfilter(b, a, x)
"""
# Second order Differential Microphone Array 2-DMA implementation
#     + weighted sum
# —————————————————————————————————————————————————————————
def beamform_2nd_order(irs, D, wL, wC, wR):
    """
    Parameters:
      irs         -- impulse response list [irL, irC, irR] left, center, right
      D           -- fractional delay (samples)
      wL, wC, wR  -- beamformer weights
    """
    irL, irC, irR = irs

    # Delay part.
    # delay + for left,  - for right.
    yL = frac_delay(irL,  +D)
    yC = irC.copy()
    yR = frac_delay(irR, -D)

    # zero padding for size equating.
    L = max(len(yL), len(yC), len(yR))
    pad = lambda v: np.pad(v, (0, L - len(v)))
    yL, yC, yR = pad(yL), pad(yC), pad(yR)

    # summing part also spatial filtering via weights found in design.
    return wL*yL + wC*yC + wR*yR


