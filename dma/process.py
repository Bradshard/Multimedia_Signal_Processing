import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# metrics
def schroeder(ir):
    E = np.cumsum(ir[::-1]**2)[::-1]
    return 10 * np.log10(E / E.max() + 1e-16)

def rt60(ir, fs):
    edc = schroeder(ir); t = np.arange(len(edc)) / fs
    i5, i35 = np.where(edc < -5)[0], np.where(edc < -35)[0]
    if not i5.size or not i35.size: return 0.0
    return (t[i35[0]] - t[i5[0]]) * (60/30)

def edt(ir, fs):
    edc = schroeder(ir); t = np.arange(len(edc)) / fs
    i0, i10 = np.where(edc < 0)[0], np.where(edc < -10)[0]
    if not i0.size or not i10.size: return 0.0
    return (t[i10[0]] - t[i0[0]]) * (60/10)

def drr(ir, fs, early_ms=50.0):
    peak = np.argmax(np.abs(ir))
    M    = int(early_ms * fs / 1000)
    ed   = np.sum(ir[peak:peak+M]**2)
    er   = np.sum(ir[peak+M:]**2)
    return np.inf if er < 1e-16 else 10 * np.log10(ed / er)


RIR_DIR  = "rirs"
OUT_DIR  = "results"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
METRICS  = os.path.join(OUT_DIR, "rir_metrics.txt")

for d in (OUT_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

fs = 48000  # should be same.


ir_ane, _ = sf.read(os.path.join(RIR_DIR, "rir_anechoic_orient000_mic1.wav"))
ir_rev, _ = sf.read(os.path.join(RIR_DIR, "rir_reverb_orient000_mic1.wav"))

plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
plt.plot(ir_ane)
plt.title("Anechoic RIR (orient000 mic1)")
plt.subplot(1,2,2)
plt.plot(ir_rev)
plt.title("Reverberant RIR (orient000 mic1)")
plt.tight_layout()
# save for cpu time.
comp_path = os.path.join(PLOT_DIR, "compare_ane_vs_rev_orient000_mic1.png")
plt.savefig(comp_path)
plt.close()


with open(METRICS, "w") as fout:
    header    = f"{'File':50s}  RT60(s)  EDT(s)   DRR(dB)\n"
    separator = "-" * 75 + "\n"
    fout.write(header + separator)
    print(header, end=""); print(separator, end="")

    # Processing rirs
    for path in sorted(glob.glob(f"{RIR_DIR}/rir_*_mic*.wav")):
        ir, _ = sf.read(path)

        #plotting 50 ms waveforms
        t = np.arange(len(ir)) / fs
        plt.figure(figsize=(4,1.5))
        plt.plot(t[:int(0.05*fs)], ir[:int(0.05*fs)])
        plt.title(os.path.basename(path))
        plt.xlabel("Time [s]")
        plt.tight_layout()
        figfile = os.path.join(PLOT_DIR,
                    os.path.basename(path).replace(".wav", ".png"))
        plt.savefig(figfile)
        plt.close()
        print(f"Saved plot: {figfile}")

        # metric results
        t60 = rt60(ir, fs)
        te = edt(ir, fs)
        dr_val = drr(ir, fs)
        dr_str = "∞" if np.isinf(dr_val) else f"{dr_val:6.2f}"

        # write to txt.
        fname = os.path.basename(path)
        line  = f"{fname:50s}  {t60:7.3f}  {te:7.3f}  {dr_str:>7s}\n"
        fout.write(line)
        print(line, end="")

print(f"\nAll metrics to {METRICS}")
print(f"All plots to the {PLOT_DIR}/")
