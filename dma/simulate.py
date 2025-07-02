import os
import numpy as np
import pyroomacoustics as pra
import soundfile as sf

def rotate_mics(base, degree_theta):
    theta = np.deg2rad(degree_theta)
    R = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [        0,         0,           1]
    ])
    center = np.array([X0, 2.0, 2.0])[:,None]
    return R @ (base - center) + center
    
    
def simulate(label, absorption, max_order):
    room = pra.ShoeBox(room_dim, fs=fs,
                       absorption=absorption,
                       max_order=max_order) # simple shoebox room for simulation
    room.add_source(source_pos)
    for orient, mpos in mic_positions.items():
        room.add_microphone_array(pra.MicrophoneArray(mpos, fs=fs))
        room.compute_rir()
        for i, rirs in enumerate(room.rir):
            rir = rirs[0] # we have one system, but enumerate sometimes confuses.
            fname = f"rir_{label}_orient{orient:03d}_mic{i+1}.wav"
            path = os.path.join(dir, fname)
            sf.write(path, rir, fs, subtype='PCM_16')
        room.mic_array = None


# Lecture-hall geometry defined (6×4×4 m), far-field > 1 m, d=2.54 cm
room_dim   = [6.0, 4.0, 4.0]    # Simple Cartesian [x,y,z] or [l, w, h] meters defined
d          = 0.0254              # 2.54 cm selected from the paper. for speech telephone band.
X0         = 3.0                 # as defined in image placed D> 1m so x0 = 3m chosen.
mic_base   = np.array([
    [X0 - d, 2.0, 2.0],          # Mic 1
    [X0    , 2.0, 2.0],          # Mic 2
    [X0 + d, 2.0, 2.0],          # Mic 3
]).T
source_pos = [5.0, 0.0, 0.0]      # source at x=5.0, same y=2.0, z=2.0
angles     = [0, 45, 90, 135]     # four orientations

fs = 48000 # 48 kHz standard sampling.
max_order = 20   # for a 500ms lecture hall enough precision to capture
dir = "rirs"
os.makedirs(dir, exist_ok=True)
labels = ("anechoic", "reverb")

# 16-bit Pulse-Code Modulation encoding
"""
PCM: raw, uncompressed audio for standard CD format as integer here in 16 bit so the range –32768 to +32767 Hence the 16 in PCM-16
"""

if __name__ == "__main__":

    mic_positions = {a: rotate_mics(mic_base, a) for a in angles}

    # Anechoic
    simulate("anechoic", absorption=None, max_order=0)

    # Reverberant(RT60≈ 500ms = 0.5 s)
    alpha, _ = pra.inverse_sabine(rt60=0.5, room_dim=room_dim) # reverberant room
    absw = {w:alpha for w in ["east","west","north","south","ceiling","floor"]}
    simulate("reverb", absorption=absw, max_order=max_order)

    # Generation of back direction as defined in the project by changing places on front for 3->2->1 swap 1 and 3. mic1<-mic3, mic2 stays, mic3<-mic1
    for label in labels:
        for i in [1, 2, 3]:
            path_in = os.path.join(dir, f"rir_{label}_orient000_mic{i}.wav")
            data, _ = sf.read(path_in)
            sf.write(os.path.join(dir, f"rir_{label}_orient180_mic{4-i}.wav"), data, fs, subtype='PCM_16')


