# Differential Microphone Array Project

This repository implements a second‑order differential microphone array (DMA) design and simulation pipeline using PyRoomAcoustics and Python. The scripts cover:

- ``: RIR generation (anechoic + reverberant)
- ``: Grid‑search for optimal cardioid coefficients & delay
- ``: Second‑order DMA beamforming (delay + filter + sum)
- ``: RIR analysis (RT60, EDT, DRR) and waveform snapshots
- ``: Polar beam patterns & speech convolution + beamforming

---

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  numpy scipy matplotlib pyroomacoustics soundfile
  ```

---


## File Structure

```
README.md         # this document
simulate.py       # Step 1: RIR generation
design.py         # Step 3: grid-search + params.json
beamformer.py     # DMA beamformer implementation
process.py        # Step 2: RIR metrics & figures
evaluate.py       # Step 4: beam patterns & speech tests
rirs/             # output RIR wavs
beamformed/       # text + JSON params & beamformed RIRs
results/          # final plots + processed speech
```

---

## Usage

1. **Generate RIRs** (anechoic + RT60≈0.5 s):
   ```bash
   python simulate.py
   ```
2. **Design beamformer** (find optimal a0,a1,a2 and delay D):
   ```bash
   python design.py
   ```
3. **Analyze RIRs** (RT60, EDT, DRR, save plots + table):
   ```bash
   python process.py
   ```
4. **Evaluate beam patterns & speech**:
   ```bash
   python evaluate.py
   ```
   - Downloads a 5 s anechoic speech sample
   - Saves `results/beam_patterns.png`
   - Saves `results/speech_beam/*.wav`

---

## Outputs

- ``: 15 anechoic + 15 reverberant RIR WAVs
- ``: `params.json`, `coeffs_for_weights_delay.txt`
- ``: polar plots, speech‑beam WAVs, beamformed WAVs

---

