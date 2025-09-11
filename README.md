# ⚠️ Important Disclaimer
This project is intended strictly for academic research inside licensed RF laboratories with proper shielding (e.g., Faraday cage).
Using signal jammers outside a controlled environment is illegal and may incur severe penalties. Always follow safety and compliance protocols.

---
# 📌 Overview

This project demonstrates how to create a basic SDR-based mobile signal jammer for controlled lab environments.
It uses GNU Radio and an SDR device (e.g., HackRF One, USRP, BladeRF) to generate wideband RF noise, which can interfere with 2G/3G/4G/5G mobile signals for testing and research purposes.
---

## 🛠 Requirements
  
# Hardware :-

  * SDR Device

  
  * HackRF One (1 MHz – 6 GHz, affordable)

  * USRP (professional research)
    
  * BladeRF / LimeSDR (alternative options)
 
    
 
* RF Amplifier (Optional)

  * To boost signal strength for more effective jamming

  * Example: Mini-Circuits ZHL series
 
    

* Wideband Antenna

  * Covers 800 MHz – 3.8 GHz for 2G–5G sub-6 GHz

  * Example: discone, log-periodic, wideband whip
 
  

* RF Shielded Lab / Faraday Cage (mandatory)

     * Prevents unauthorized interference outside the lab


# Software :-

  * GNU Radio (signal processing framework)

  * SDR Drivers & APIs
       * gr-osmosdr
       * Vendor-specific drivers (HackRF, USRP, etc.)

  * Python 3.x
  * Spectrum Analyzer Software
       * e.g., SDR#, GNU Radio Companion

# Knowledge Areas

  * Wireless communication protocols (2G–5G frequency bands)
  * DSP (Digital Signal Processing)
  * RF engineering concepts
  * Lab safety & compliance

# 📡 Supported Frequency Bands

| Technology | Frequency Bands (Typical)    |
| ---------- | ---------------------------- |
| 2G (GSM)   | 900 / 1800 MHz               |
| 3G (UMTS)  | 2100 MHz                     |
| 4G (LTE)   | 800 / 1800 / 2100 / 2600 MHz |
| 5G (NR)    | 3.3–3.8 GHz (sub-6 GHz)      |

# ⚠️ Note: HackRF One does not support 5G mmWave (>6 GHz).

---
## 🚀 Setup & Usage

# Step 1: Install Dependencies

On Ubuntu/Debian:- 
           bash 
               * sudo apt-get update
sudo apt-get install gnuradio gr-osmosdr hackrf


# Step 2: Connect Hardware :- 

csharp
* [Computer/PC]
      |
      | USB 3.0
      v
   [SDR Device]
      |
      | SMA Cable
      v
 [RF Amplifier] (optional)
      |
      v
   [Antenna]
      |
      v
[Inside Faraday Cage]

---

# Step 3: Run Example Jammer :-
  bash
    python3 jammer.py

* Default: jams GSM900 (900 MHz)
* Adjust { center_freq, samp_rate, and gain in jammer.py for other bands.}

---

  # (inside code):
  ```
"""
simulated_interference_analysis.py

Safe, educational simulation and analysis of a signal-of-interest under interference.
- Generates a clean signal (narrowband tone).
- Adds broadband noise and one or more intermittent narrowband interferers (bursts).
- Computes spectrograms, bandpass filtering, SNR estimates.
- Runs a simple detector that flags interference events by energy thresholding in the interferer band.
- Plots results and saves figures.

Requirements:
pip install numpy scipy matplotlib

Run:
python simulated_interference_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, welch
import os

# --------------------------
# Parameters
# --------------------------
fs = 48000.0          # sampling rate (Hz)
duration = 5.0        # total signal duration (seconds)
t = np.arange(0, duration, 1/fs)

# Signal of interest (narrowband tone)
f_sig = 1200.0        # Hz
amp_sig = 1.0

# Interferer(s) parameters (narrowband tones, intermittent bursts)
interferers = [
    {"f": 3000.0, "amp": 2.0, "start": 1.0, "stop": 1.6},   # a burst from 1.0s to 1.6s
    {"f": 3200.0, "amp": 1.2, "start": 3.2, "stop": 3.9}    # another burst later
]

# Broadband noise level
noise_std = 0.3

# Detector settings
interf_band = (2800.0, 3400.0)    # frequency band to monitor for interference (Hz)
detection_window = 0.05           # seconds per detection block
energy_threshold_db = -10.0       # dB relative threshold for detection (tunable)

# Output directory
outdir = "sim_analysis_outputs"
os.makedirs(outdir, exist_ok=True)

# --------------------------
# Generate signals
# --------------------------
# Clean signal
signal = amp_sig * np.sin(2*np.pi*f_sig*t)

# Interferers: create time-limited tone bursts
interferer_signal = np.zeros_like(t)
for it in interferers:
    m = (t >= it["start"]) & (t <= it["stop"])
    # optional smooth window for burst edges
    win = np.ones_like(t[m])
    # apply a Hanning ramp of 10 ms at start/stop to reduce abrupt edges
    ramp_ms = 10e-3
    ramp_samples = int(round(ramp_ms * fs))
    if ramp_samples > 0 and len(win) > 2*ramp_samples:
        r = np.hanning(2*ramp_samples*2)[0:2*ramp_samples]  # just to create a smooth shape
        # linear create small fade-in/out
        fade_in = np.hanning(2*ramp_samples)[:ramp_samples]
        fade_out = np.hanning(2*ramp_samples)[ramp_samples:]
        win[:ramp_samples] = fade_in
        win[-ramp_samples:] = fade_out
    interferer_signal[m] += it["amp"] * np.sin(2*np.pi*it["f"]*t[m]) * win

# Broadband noise
noise = noise_std * np.random.normal(size=t.shape)

# Observed signal combines everything
observed = signal + interferer_signal + noise

# --------------------------
# Helper functions
# --------------------------
def bandpass_filter(x, fs, lowcut, highcut, order=4):
    nyq = fs/2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, x)

def compute_snr(signal_band, noise_band):
    # Estimate SNR in dB using power in signal band vs power outside it (approximate)
    # signal_band and noise_band are signals (time-domain)
    p_sig = np.mean(signal_band**2)
    p_n = np.mean(noise_band**2) + 1e-15
    return 10*np.log10(p_sig / p_n)

# --------------------------
# Analysis 1: spectrogram
# --------------------------
nperseg = 2048
noverlap = nperseg // 2
f, ts, Sxx = spectrogram(observed, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
Sxx_db = 20*np.log10(Sxx + 1e-12)

plt.figure(figsize=(10, 5))
plt.pcolormesh(ts, f, Sxx_db, shading='gouraud')
plt.colorbar(label='Magnitude (dB)')
plt.ylim(0, 6000)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of observed signal')
plt.tight_layout()
plt.savefig(os.path.join(outdir, "spectrogram_observed.png"))
plt.close()

# --------------------------
# Analysis 2: bandpass filter around signal of interest
# --------------------------
sig_bandwidth = 200.0  # +/- around f_sig
sig_low = f_sig - sig_bandwidth/2
sig_high = f_sig + sig_bandwidth/2
filtered_sig = bandpass_filter(observed, fs, sig_low, sig_high)

# Also bandpass for interferer monitoring band
i_low, i_high = interf_band
filtered_interf = bandpass_filter(observed, fs, i_low, i_high)

# Time domain plot (snippet)
snippet_samples = int(0.05 * fs)  # 50 ms
start_idx = int(0.5*fs)  # show around 0.5s
time_snip = t[start_idx:start_idx+snippet_samples]

plt.figure(figsize=(10,4))
plt.plot(time_snip, observed[start_idx:start_idx+snippet_samples], label='Observed', alpha=0.6)
plt.plot(time_snip, filtered_sig[start_idx:start_idx+snippet_samples], label='Filtered around signal', linewidth=1.2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain (50 ms snippet)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "time_snippet.png"))
plt.close()

# --------------------------
# Analysis 3: SNR estimation over time (sliding blocks)
# --------------------------
block_size = int(detection_window * fs)
n_blocks = len(t) // block_size
times_blocks = (np.arange(n_blocks) * block_size) / fs
snr_estimates = np.zeros(n_blocks)
interf_energy_db = np.zeros(n_blocks)

for k in range(n_blocks):
    s_idx = k * block_size
    e_idx = s_idx + block_size
    block = observed[s_idx:e_idx]
    # Estimate power in signal-band and in interference-band via bandpass filtering
    sig_block = bandpass_filter(block, fs, sig_low, sig_high) if len(block) > 10 else np.zeros_like(block)
    interf_block = bandpass_filter(block, fs, i_low, i_high) if len(block) > 10 else np.zeros_like(block)
    # compute energies
    p_sig = np.mean(sig_block**2) + 1e-15
    p_interf = np.mean(interf_block**2) + 1e-15
    # For SNR: treat everything else as noise approx: observed power - signal power
    p_obs = np.mean(block**2) + 1e-15
    p_noise_est = max(p_obs - p_sig, 1e-15)
    snr_estimates[k] = 10*np.log10(p_sig / p_noise_est)
    interf_energy_db[k] = 10*np.log10(p_interf)

# --------------------------
# Detection: flag blocks where interferer energy exceeds threshold relative to median
# --------------------------
median_interf_energy = np.median(interf_energy_db)
threshold_abs = median_interf_energy + abs(energy_threshold_db)  # relative offset (tune as required)
detections = interf_energy_db > threshold_abs

# Create a simple list of detected intervals (start_time, end_time)
detected_intervals = []
inside = False
for k, det in enumerate(detections):
    if det and not inside:
        start_time = times_blocks[k]
        inside = True
    elif not det and inside:
        end_time = times_blocks[k]
        detected_intervals.append((start_time, end_time))
        inside = False
if inside:
    detected_intervals.append((start_time, times_blocks[-1] + detection_window))

# --------------------------
# Plots: SNR and interferer energy with detection markers
# --------------------------
plt.figure(figsize=(10,4))
plt.plot(times_blocks, snr_estimates, label='Estimated SNR (dB)')
plt.xlabel('Time (s)')
plt.ylabel('SNR (dB)')
plt.title('Estimated SNR over time (block size {:.1f} ms)'.format(detection_window*1000))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "snr_over_time.png"))
plt.close()

plt.figure(figsize=(10,4))
plt.plot(times_blocks, interf_energy_db, label='Interferer band energy (dB)')
plt.axhline(threshold_abs, color='r', linestyle='--', label='Detection threshold')
# overlay detected blocks
for interval in detected_intervals:
    plt.axvspan(interval[0], interval[1], color='red', alpha=0.2)
plt.xlabel('Time (s)')
plt.ylabel('Energy (dB)')
plt.title('Interferer band energy and detections')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "interf_energy_and_detections.png"))
plt.close()

# --------------------------
# PSD (Welch) for Observed vs Clean signal
# --------------------------
f_obs, P_obs = welch(observed, fs=fs, nperseg=4096)
f_clean, P_clean = welch(signal, fs=fs, nperseg=4096)

plt.figure(figsize=(10,4))
plt.semilogy(f_obs, P_obs, label='Observed (signal + interferers + noise)')
plt.semilogy(f_clean, P_clean, label='Clean signal only', alpha=0.8)
plt.xlim(0, 6000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Welch PSD: Observed vs Clean')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "psd_welch.png"))
plt.close()

# --------------------------
# Summary printed output
# --------------------------
print("Simulation complete. Outputs saved to folder:", outdir)
print("Detected interference intervals (approx):")
if detected_intervals:
    for idx, (s_i, e_i) in enumerate(detected_intervals, start=1):
        print(f"  {idx}. {s_i:.3f}s -> {e_i:.3f}s")
else:
    print("  None detected with current threshold settings.")
print()
print("Tweak parameters at the top of the script (interferers, detection threshold, detection window) to explore behavior.")

# Optionally: save some arrays for further analysis
np.savez(os.path.join(outdir, "sim_data.npz"),
         t=t, observed=observed, signal=signal, interferer_signal=interferer_signal,
         snr_estimates=snr_estimates, interf_energy_db=interf_energy_db,
         detected_intervals=np.array(detected_intervals, dtype=object))

```

---

## 🧪 Research Workflow

  1. Identify target frequency band

  2. Configure SDR parameters

  3. Generate noise/interference signal

  4.(Optional) Amplify with RF amplifier

  5. Transmit via antenna inside shielded lab

  6. Monitor with spectrum analyzer & mobile devices

  7.Document results (range, effectiveness, side-effects)

 --- 

## 🔒 Safety & Legal Compliance

  * Operate only in RF-shielded labs with proper licenses

  * RF energy can be hazardous—follow lab safety protocols

  * Never operate outdoors or in public spectrum

  * Confirm all experiments with your supervisor

---

## 📚 Useful Resources

   * GNU Radio Tutorials
   * HackRF One Documentation
   * USRP Getting Started
   * Mobile Network Frequency Bands

---

## 📊 Summary Table


| Component         | Purpose                       | Example/Source        |
| ----------------- | ----------------------------- | --------------------- |
| SDR Hardware      | Generate RF signals           | HackRF, USRP, BladeRF |
| RF Amplifier      | Boost signal power            | Mini-Circuits ZHL     |
| Antenna           | Transmit jamming signal       | Wideband antenna      |
| GNU Radio         | Signal processing & control   | gnuradio.org          |
| Spectrum Analyzer | Monitor jamming effectiveness | SDR#, GNU Radio       |

---

## ⚠️ Ethical Note

This project is intended for educational and research purposes only.
Unauthorized signal jamming is illegal, dangerous, and unethical.

---
  
