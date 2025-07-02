import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from beamformer import beamform_2nd_order
from scipy.signal import resample, butter, filtfilt
import urllib.request
plt.switch_backend('Agg')

def align_rir(rir, thresh=1e-6):
    n0 = np.argmax(np.abs(rir) > thresh)
    return np.roll(rir, -n0)

# load speech file
speech_url  = "https://webfiles.york.ac.uk/OPENAIR/Anechoic/speech-macbeth/Speech%20-%20Core%20Take.wav"
speech_file = "anechoic_speech.wav"

if not os.path.exists(speech_file):
    print(f"Downloading speech sample to '{speech_file}' …")
    urllib.request.urlretrieve(speech_url, speech_file)
    print("Download complete.")

# --- Load DMA params and constants with directories---
with open("beamformed/params.json") as f:
    p = json.load(f)
D, wL, wC, wR = p["D"], p["wL"], p["wC"], p["wR"]
a0,a1,a2 = p["a0"], p["a1"], p["a2"]


fs = 48000
oris = [0, 45, 90, 135, 180]
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# Calculate and plot directional beam patterns (anechoic RIRs)
# FFT setup
Nfft  = 16384
faxis = rfftfreq(Nfft, 1/fs)

# retrieval of beamformed magnitude
B = []
for ori in oris:
    # anechoic RIRs for each orientation
    irs = [sf.read(f"rirs/rir_anechoic_orient{ori:03d}_mic{i}.wav")[0] for i in (1,2,3)]
    bf = beamform_2nd_order(irs, D, wL, wC, wR)
    H = np.abs(rfft(bf, Nfft))
    B.append(H)
B = np.stack(B)  # shape (5, len(faxis))

freqs = [100,7200, 48000]  # Hz
thetas_rad  = np.deg2rad(oris)
eps = 1e-3 # for to avoud error at 0.

plt.figure(figsize=(12,4))
for i, f in enumerate(freqs):
    # find the closest FFT bin
    idx  = np.argmin(np.abs(faxis - f))
    mags = 20 * np.log10(B[:, idx] / B[:, idx].max())

    ax = plt.subplot(1, 3, i+1, projection='polar')
    ax.plot(thetas_rad, mags, 'o-', label='simulated')

    # theoretical 2nd-order cardioid (clipped)
    th  = np.linspace(0, 2*np.pi, 360)
    Dth = a0 + a1*np.cos(th) + a2*np.cos(2*th)
    Dth_clip = np.clip(Dth, eps, None)
    ax.plot(th, 20 * np.log10(Dth_clip), '--', label='theoretical')

    ax.set_title(f"{f} Hz")
    if i == 0:
        ax.set_ylim(-40, 0) # floor.
        ax.legend(loc='lower right')

plt.suptitle("Simulated vs. Theoretical Cardioid Patterns")
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(out_dir, "beam_patterns.png"))
plt.close()

# Convolve reverberant RIRs with speech & beamform
# load speech
speech_full, fs_s = sf.read(speech_file)
# actually fs_s = 48000 so no need but still.

if fs_s != 48000:
    num = int(len(speech)*48000/fs_s)
    speech = resample(speech, num)
    fs_s = 48000

# keep only the first 5 seconds
N_trim = int(fs_s * 5)
speech = speech_full[:N_trim]

print(f"Using {len(speech)/fs_s:.1f}s of speech @ {fs_s}Hz")

# now the 5 s speech is only ~0.7 MB instead of ~18 MB
# design High Pass filter
#hp_b, hp_a = butter(2, 100/(fs_s/2), btype='high')

bf_dir = os.path.join(out_dir, "speech_beam")
os.makedirs(bf_dir, exist_ok=True)
oris = [0, 45, 90, 135, 180]
for ori in oris:
    # alignment and loading of mic's revb RIR
    rirs = []
    for i in (1,2,3):
        rir, _ = sf.read(f"rirs/rir_reverb_orient{ori:03d}_mic{i}.wav")
        rirs.append(align_rir(rir))

    
    chs = [np.convolve(speech, r) for r in rirs]# convolve speech w/ aligned RIR
    bf_sig = beamform_2nd_order(chs, D, wL, wC, wR) # Beamform 2nd order (delay+filter+sum)

    # high-pass to remove weird parts
    #bf_sig = filtfilt(hp_b, hp_a, bf_sig)
    #bf_sig /= np.max(np.abs(bf_sig)) + 1e-12 # normalize for clearer results
    out_path = os.path.join(bf_dir, f"speech_beam_rev_orient{ori:03d}.wav") # save to proper folder.
    sf.write(out_path, bf_sig, fs_s, subtype="PCM_16")

print(f" Beam patterns sent to {out_dir}/beam_patterns.png")
print(f" Speech‐beamformed to {bf_dir}/")

