"""
Configuration parameters and default values for TEP features extraction.
"""

# Default temporal windows (in seconds) for common TEP peaks
# Based on TEP literature.
DEFAULT_WINDOWS = {
    "N15": [0.010, 0.020],
    "P30": [0.025, 0.035],
    "N45": [0.035, 0.055],
    "P60": [0.055, 0.075],
    "N100": [0.090, 0.125],
    "P200": [0.150, 0.220]
}

# Default frequency bands (in Hz) for spectral analysis
DEFAULT_BANDS = {
    "alpha": [8.0, 12.0],
    "beta": [13.0, 30.0],
}
