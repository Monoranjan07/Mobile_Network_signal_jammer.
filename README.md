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

# 📡 Supported Frequency Bands

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


# Step 3: Run Example Jammer :-
  bash
    python3 jammer.py

* Default: jams GSM900 (900 MHz)
* Adjust { center_freq, samp_rate, and gain in jammer.py for other bands.}


  # Example (inside code):
     '''


     '''


## 🧪 Research Workflow

  1. Identify target frequency band

  2. Configure SDR parameters

  3. Generate noise/interference signal

  4.(Optional) Amplify with RF amplifier

  5. Transmit via antenna inside shielded lab

  6. Monitor with spectrum analyzer & mobile devices

  7.Document results (range, effectiveness, side-effects)

  

## 🔒 Safety & Legal Compliance

  * Operate only in RF-shielded labs with proper licenses

  * RF energy can be hazardous—follow lab safety protocols

  * Never operate outdoors or in public spectrum

  * Confirm all experiments with your supervisor


## 📚 Useful Resources

   * GNU Radio Tutorials
   * HackRF One Documentation
   * USRP Getting Started
   * Mobile Network Frequency Bands

## 📊 Summary Table


| Component         | Purpose                       | Example/Source        |
| ----------------- | ----------------------------- | --------------------- |
| SDR Hardware      | Generate RF signals           | HackRF, USRP, BladeRF |
| RF Amplifier      | Boost signal power            | Mini-Circuits ZHL     |
| Antenna           | Transmit jamming signal       | Wideband antenna      |
| GNU Radio         | Signal processing & control   | gnuradio.org          |
| Spectrum Analyzer | Monitor jamming effectiveness | SDR#, GNU Radio       |


## ⚠️ Ethical Note

This project is intended for educational and research purposes only.
Unauthorized signal jamming is illegal, dangerous, and unethical.


  
