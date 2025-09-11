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
# 1. Receive-only SoapySDR capture (safe)

  * This captures IQ samples from the first available SDR device and writes them to a file. No transmit.
Requirements: SoapySDR, numpy.

```
# soapy_receive_capture.py
# Receive-only IQ capture (safe, does not transmit)
# Requirements: SoapySDR, numpy
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import sys
import time

CENTER_FREQ = 900e6    # change to a frequency you are permitted to observe
SAMPLE_RATE = 1e6      # 1 MS/s
GAIN = 30
DURATION_SEC = 5      # seconds to capture
OUTFILE = "iq_capture.bin"  # raw complex64 interleaved (I,Q)

def main():
    devices = SoapySDR.Device.enumerate()
    if not devices:
        print("No SDR devices found. Connect an SDR and try again.")
        return

    dev = SoapySDR.Device(devices[0])
    rx_chan = 0

    dev.setSampleRate(SOAPY_SDR_RX, rx_chan, SAMPLE_RATE)
    dev.setFrequency(SOAPY_SDR_RX, rx_chan, CENTER_FREQ)
    dev.setGain(SOAPY_SDR_RX, rx_chan, GAIN)

    rx_stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [rx_chan])
    dev.activateStream(rx_stream)

    total_samples = int(SAMPLE_RATE * DURATION_SEC)
    buf_size = 4096
    written = 0

    print(f"Capturing {DURATION_SEC} s ({total_samples} samples) at {CENTER_FREQ/1e6} MHz -> {OUTFILE}")
    with open(OUTFILE, "wb") as f:
        while written < total_samples:
            to_read = min(buf_size, total_samples - written)
            buff = np.array([0j]*to_read, dtype=np.complex64)
            sr = dev.readStream(rx_stream, [buff], to_read)
            if sr.ret > 0:
                # write interleaved complex64 (float32 real, float32 imag)
                buff.tofile(f)
                written += sr.ret
            else:
                print("readStream error or timeout:", sr.ret)
                time.sleep(0.01)

    dev.deactivateStream(rx_stream)
    dev.closeStream(rx_stream)
    print("Capture complete.")

if _name_ == "_main_":
    main()
```

 * How to use: run it in a controlled environment. The file iq_capture.bin can be analyzed offline (plot spectrum, compute PSD, demodulate, etc.).

---

# 2. Simulation-only: generate IQ data + add noise (no hardware)

   * Generates a QPSK-like signal, applies interference/noise, and saves IQ samples. Great for studying constellation, BER, and spectrums — no RF.

```
# simulate_iq_and_noise.py
# Generates QPSK symbols, adds noise (simulated jammer), saves IQ samples to disk.
# Requirements: numpy, matplotlib (optional for plotting)

import numpy as np
import matplotlib.pyplot as plt

NUM_SYMBOLS = 20000
SAMPLE_RATE = 1e6  # for reference only
SNR_DB = 10        # signal-to-noise ratio in dB
OUTFILE = "simulated_iq.bin"

# QPSK mapping
bits = np.random.randint(0, 4, NUM_SYMBOLS)
symbols = np.exp(1j * (np.pi/2 * bits + np.pi/4))  # Gray-coded QPSK
# Upsample / shape pulse if needed; here we use symbol-rate IQ simple model
signal = symbols.astype(np.complex64)

# Add Gaussian noise
snr_linear = 10**(SNR_DB/10)
signal_power = np.mean(np.abs(signal)**2)
noise_power = signal_power / snr_linear
noise = (np.random.randn(NUM_SYMBOLS) + 1j*np.random.randn(NUM_SYMBOLS)) * np.sqrt(noise_power/2)
rx = (signal + noise).astype(np.complex64)

# Save to file (interleaved float32: I,Q,I,Q,...)
rx.tofile(OUTFILE)
print(f"Saved simulated IQ to {OUTFILE}")

# Optional: quick plots
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(np.real(rx[:2000]), np.imag(rx[:2000]), s=2)
plt.title("Constellation (subset)")
plt.subplot(1,2,2)
plt.psd(rx, NFFT=1024, Fs=SAMPLE_RATE/1e3) # frequency in kHz on axis
plt.title("PSD (simulated)")
plt.tight_layout()
plt.show()
```

* This file is safe to share in labs and to analyze in software suites.
  ---

  # 3. Local network "transmitter" (UDP) — learn streaming mechanics (safe)

This sends arbitrary data over your LAN using UDP. Useful for practicing streaming, latency, packet loss, and protocols without touching RF.

* * Sender (transmitter):
```
# udp_sender.py
import socket
import time

DEST_IP = "127.0.0.1"   # loopback or target host on LAN
DEST_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = b"Hello, this is a safe UDP transmission test."

for i in range(100):
    sock.sendto(message + f" #{i}".encode(), (DEST_IP, DEST_PORT))
    time.sleep(0.1)

print("Done sending.")

```
---

* * Receiver:
```
 udp_receiver.py
import socket

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))
print(f"Listening on {LISTEN_IP}:{LISTEN_PORT}")

while True:
    data, addr = sock.recvfrom(4096)
    print("Received:", data, "from", addr)

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
  
