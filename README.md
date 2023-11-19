# Enhancing Aircraft Electronic Warfare Testing with Automated RF Spectrum Analysis

**Project Overview:**

Military flight test ranges play a crucial role in the development and validation of Aircraft Electronic Warfare (EW) Systems. These ranges utilize a variety of Radio Frequency (RF) Threat Systems, both real and simulated, to assess the effectiveness of these systems during flight tests. A pivotal component of this process involves monitoring and recording RF transmissions, which are essential for calibration and verification. Traditionally, system engineers at Robins Airforce Base have manually analyzed video data from Spectrum Analyzers to confirm the frequency and amplitude of specific Threat Systems.

**Project Objective:**

To streamline and enhance this critical analysis, our project aimed to develop an automated solution for RF Spectrum Analysis. We employed a custom trained state-of-the-art YOLO V8 model to detect and isolate the Spectrum Analyzer screen in any arbitrary video footage regardless of the analyzer used. Furthermore, we utilized a novel combination of frame differencing, summing, and agglomerative clustering techniques to isolate and segment the signals measured on the spectrum analyzer screen. In this way, relevant properties of the measured signals could be extracted from the analyzed video. Finally, to maximize video processing speed, we used Python’s multiprocessing library to parallelize the processing of frame batches and to leverage all available processor cores.

**Key Features:**

- Detection and isolation of Spectrum Analyzer screens in diverse video footage.
- Automated extraction of signal amplitude and frequency.
- Identification of a measured signal’s center frequency and peak amplitude.

**Running the Application**
First install any missing dependencies:
```bash
pip install -r requirements.txt
```
Run interface.py:
```bash
python interface.py
```

**Authors:**

- Anthony De Santiago - Project Lead - [@AnthonyDeSantiago](https://github.com/AnthonyDeSantiago)
- Camille Reaves - [@Camille Reaves](https://github.com/camillereaves)
- Mathew Morgan - [@Hurkaleez](https://github.com/Hurkaleez)
- Geonhyeong Kim - [@Davifus](https://github.com/Davifus)
- Jalon Bailey - [@jalon360](https://github.com/jalon360)

---
