# ⚠️ Important Disclaimer
This project is intended strictly for academic research inside licensed RF laboratories with proper shielding (e.g., Faraday cage).
Using signal jammers outside a controlled environment is illegal and may incur severe penalties. Always follow safety and compliance protocols.


# 📌 Overview

This project demonstrates how to create a basic SDR-based mobile signal jammer for controlled lab environments.
It uses GNU Radio and an SDR device (e.g., HackRF One, USRP, BladeRF) to generate wideband RF noise, which can interfere with 2G/3G/4G/5G mobile signals for testing and research purposes.

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

## 📡 Supported Frequency Bands

| Technology | Frequency Bands (Typical)    |
| ---------- | ---------------------------- |
| 2G (GSM)   | 900 / 1800 MHz               |
| 3G (UMTS)  | 2100 MHz                     |
| 4G (LTE)   | 800 / 1800 / 2100 / 2600 MHz |
| 5G (NR)    | 3.3–3.8 GHz (sub-6 GHz)      |

# ⚠️ Note: HackRF One does not support 5G mmWave (>6 GHz).


## 🚀 Setup & Usage

# Step 1: Install Dependencies

On Ubuntu/Debian:- 
           bash 
               sudo apt-get update
sudo apt-get install gnuradio gr-osmosdr hackrf


# Step 2: Connect Hardware :- 

csharp         
  [Computer/PC]
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
