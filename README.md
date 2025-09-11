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

# ⚠️ Note:- My dear friend, use your brain a little again, because here are 3 Python codes, understand them well and implement them 🤯
---
```
"""
Safe OFDM + Interferer IQ Generator (No Transmission)
Author: assistant (for Monoranjan07 project)

Generates an OFDM waveform and overlays selectable interference.
Saves interleaved complex64 IQ to 'simulated_jam_iq.bin'.

Requirements:
    pip install numpy matplotlib scipy

Use: analyze offline, load into lab instrumentation, or feed to receive-only SDR capture/analyzer.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# -----------------------
# Parameters
# -----------------------
fs = 1e6                 # sample rate (Hz) for simulation (sample-rate is for file metadata only)
num_subcarriers = 256    # OFDM subcarriers
cp_len = 64              # cyclic prefix length (samples)
num_ofdm_symbols = 400   # number of OFDM symbols
mod_order = 4            # QPSK (2 bits/symbol)
subcarrier_spacing = fs / num_subcarriers  # Hz
output_file = "simulated_jam_iq.bin"

# Interferer options (choose one or combine)
add_tone = True
tone_freq_hz = 50e3      # tone frequency offset from center (Hz) — e.g., 50 kHz offset
tone_amplitude_db = -3   # relative to OFDM (dB)
tone_pulsed = True       # if True, tone pulses on/off
tone_pulse_period = 0.02 # seconds (period of on+off)
tone_pulse_duty = 0.5    # fraction of period tone is ON

add_narrowband_noise = False
nb_noise_amplitude_db = -6  # dB

add_wideband_noise = False
wb_snr_db = 10

# -----------------------
# Helper functions
# -----------------------
def qpsk_mod(bits):
    b = bits.reshape((-1,2))
    sym = (2*b[:,0]-1) + 1j*(2*b[:,1]-1)
    return sym / np.sqrt(2)

def generate_ofdm_frame(num_subcarriers, cp_len, data_symbols):
    # data_symbols length == num_subcarriers
    tx_ifft = np.fft.ifft(data_symbols, n=num_subcarriers)
    tx_with_cp = np.hstack([tx_ifft[-cp_len:], tx_ifft])
    return tx_with_cp

# -----------------------
# Build OFDM waveform
# -----------------------
ofdm_time_waveform = []

for _ in range(num_ofdm_symbols):
    # random bits for each subcarrier (QPSK)
    bits = np.random.randint(0,2, num_subcarriers*2)
    qpsk = qpsk_mod(bits)
    tx_sym = generate_ofdm_frame(num_subcarriers, cp_len, qpsk)
    ofdm_time_waveform.append(tx_sym)

ofdm_time = np.hstack(ofdm_time_waveform).astype(np.complex64)
num_samples = ofdm_time.size
duration_sec = num_samples / fs
print(f"Generated OFDM waveform: {num_samples} samples, duration ~ {duration_sec:.3f} s")

# normalize OFDM to unit RMS
ofdm_rms = np.sqrt(np.mean(np.abs(ofdm_time)**2))
ofdm_time = ofdm_time / ofdm_rms

# -----------------------
# Build interferer
# -----------------------
t = np.arange(num_samples) / fs
interferer = np.zeros(num_samples, dtype=np.complex64)

if add_tone:
    # tone as complex baseband: exp( j*2pi*freq*t )
    tone_freq = tone_freq_hz  # Hz offset from center
    tone = np.exp(1j*2*np.pi*tone_freq*t)
    # amplitude relative to OFDM RMS
    amp_linear = 10**(tone_amplitude_db/20)
    tone = tone * amp_linear

    if tone_pulsed:
        # create pulsing window
        period_samples = int(tone_pulse_period * fs)
        on_samples = int(period_samples * tone_pulse_duty)
        window = np.zeros(num_samples)
        for start in range(0, num_samples, period_samples):
            window[start : start+on_samples] = 1.0
        tone = tone * window

    interferer += tone.astype(np.complex64)

if add_narrowband_noise:
    # narrowband colored noise centered at tone_freq_hz (simulate narrowband jammer)
    # generate white noise then filter in frequency domain by Gaussian around tone_freq
    noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)).astype(np.complex64)
    # frequency-domain shaping
    freqs = np.fft.fftfreq(num_samples, d=1/fs)
    center = tone_freq_hz
    sigma = 5e3  # Hz: narrowband width
    shape = np.exp(-0.5 * ((freqs - center)/sigma)**2)
    noise_fd = np.fft.fft(noise) * shape
    nb_noise = np.fft.ifft(noise_fd)
    amp = 10**(nb_noise_amplitude_db/20)
    interferer += (nb_noise * amp).astype(np.complex64)

if add_wideband_noise:
    # wideband AWGN added to whole band to reach desired SNR vs OFDM
    wb_noise = (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)).astype(np.complex64)
    # compute scaling for SNR
    wb_noise = wb_noise / np.sqrt(np.mean(np.abs(wb_noise)**2))
    amp = 10**(-wb_snr_db/20)
    interferer += (wb_noise * amp).astype(np.complex64)

# -----------------------
# Combine and normalize
# -----------------------
combined = ofdm_time + interferer
combined_rms = np.sqrt(np.mean(np.abs(combined)**2))
combined = combined / combined_rms  # normalize to 0 dBFS reference

# -----------------------
# Save to file (interleaved float32: I,Q,I,Q,...)
# -----------------------
combined.astype(np.complex64).tofile(output_file)
print(f"Saved simulated IQ to: {output_file}")

# -----------------------
# Quick plots for inspection
# -----------------------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(np.real(combined[:2000]))
plt.title("Time domain (real) - first 2000 samples")

plt.subplot(3,1,2)
# PSD via welch
f, Pxx = welch(combined, fs=fs, nperseg=4096)
plt.semilogy(f - fs/2, np.fft.fftshift(Pxx))
plt.title("Power Spectral Density (shifted)")
plt.xlabel("Frequency (Hz)")

plt.subplot(3,1,3)
# Constellation sample
sample_for_const = combined[:num_subcarriers*4]  # choose several OFDM symbols worth
plt.scatter(np.real(sample_for_const), np.imag(sample_for_const), s=2, alpha=0.6)
plt.title("Constellation (subset of samples)")
plt.xlabel("I")
plt.ylabel("Q")

plt.tight_layout()
plt.show()
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
  
