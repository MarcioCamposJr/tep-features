"""
Configuration parameters and default values for TEP features extraction.
"""

# Default temporal windows (in seconds) for common TEP peaks
# Based on TEP literature.
DEFAULT_WINDOWS = {
    "N15": [0.010, 0.020],
    "P30": [0.025, 0.035],
    "N45": [0.040, 0.055],
    "P60": [0.050, 0.070],
    "N100": [0.100, 0.160],
    "P200": [0.180, 0.280],
}
