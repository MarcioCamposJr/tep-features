# TEPFeatures 🧠⚡

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![MNE-Python](https://img.shields.io/badge/mne--python-%3E%3D1.12.1-success)
![Pandas](https://img.shields.io/badge/pandas-%3E%3D3.0.5-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A modern, fast, and structured Python library for extracting advanced features from **TMS-Evoked Potentials (TEPs)** using EEG data. Built on top of `mne-python` and `pandas`, `tepfeatures` acts as a powerful facade to automate the extraction of temporal, spatial, dynamic (microstates), and spectral features from TEPs.

---

## 🎯 Features

- **Temporal Features:** Intelligent peak extraction based on polarity, morphology descriptors (peak-to-peak amplitude, inter-peak interval, slope), and Area Under the Curve (AUC) calculated via trapezoidal integration.
- **Spatial Features:** Global Field Power (GFP), Local Mean Field Power (LMFP), automated extraction of GFP peaks within dynamic time windows, and Topographical Cosine Similarity.
- **Dynamic Features (Microstates):** Built-in modified K-Means clustering algorithm to identify periods of stable topography, extracting onset time, duration, and associated mean GFP.
- **Spectral Features:** Frequency-domain analysis utilizing Welch's method for calculating power within specific frequency bands (e.g., Alpha, Beta) and identifying the natural/dominant peak frequency.
- **Dynamic Windows:** Easily access and modify clinical standard TEP windows (e.g., N15, P30, N45, P60, N100, P200) using dict-like syntax.

## 📦 Installation

This project is built and managed with `uv`. To install it from source:

```bash
git clone https://github.com/your-username/tepfeatures.git
cd tepfeatures
uv sync
```

*Or via PyPI (coming soon!):*
```bash
pip install tepfeatures
```

## 🚀 Quickstart

```python
import mne
from tepfeatures.core import TEPExtractor

# 1. Load your TEP data (must be an mne.Evoked object)
evoked = mne.read_evokeds('your_tep_data-ave.fif')[0]

# 2. Initialize the extractor
extractor = TEPExtractor(evoked)

# You can dynamically modify standard windows if needed:
extractor['N15'] = [0.012, 0.018]

# 3. Extract Temporal Features
df_peak = extractor.temporal.get_peak('N100', channels=['Cz', 'Fz'])
df_auc = extractor.temporal.get_area('P30', channels='Cz')

# 4. Extract Spatial Features
times, gfp = extractor.spatial.get_gfp()
df_topo_sim = extractor.spatial.topographical_similarity('N15', 'P30')

# 5. Extract Spectral Features
df_bands = extractor.spectral.get_band_power() # Alpha and Beta by default
df_natural_freq = extractor.spectral.get_peak_frequency(channels='Cz')

# 6. Extract Microstates
extractor.microstates.fit(n_states=4)
df_microstates = extractor.microstates.get_features()

print(df_microstates)
```

## 📖 Documentation

The full documentation is built with MkDocs. To view it locally:

```bash
uv run mkdocs serve
```
Then open `http://127.0.0.1:8000/` in your browser.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
