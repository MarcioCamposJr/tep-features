import numpy as np
import pandas as pd
import mne
from typing import List, Union, Dict, Any, Optional

class SpectralFeatures:
    """
    Class for extracting frequency-domain (spectral) features from TMS-Evoked Potentials (TEPs).
    
    This class utilizes Power Spectral Density (PSD) to calculate band power
    and peak natural frequency.
    
    Parameters
    ----------
    extractor : TEPExtractor
        The parent extractor instance containing the Evoked data and windows.
    """
    
    def __init__(self, extractor: Any):
        self._extractor = extractor

    def get_band_power(self, bands: Optional[Dict[str, List[float]]] = None, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Calculate the power within specific frequency bands using Welch's method.
        
        Parameters
        ----------
        bands : dict, optional
            Dictionary containing band names as keys and [fmin, fmax] as values.
            Default is {'alpha': [8, 12], 'beta': [13, 30]}.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'band_name', and 'power' (V^2/Hz).
        """
        if bands is None:
            bands = {'alpha': [8, 12], 'beta': [13, 30]}
            
        evoked = self._extractor.evoked
        
        if channels is None:
            picks = mne.pick_types(evoked.info, eeg=True)
            ch_names = [evoked.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            picks = mne.pick_channels(evoked.ch_names, ch_names)
            
        # Compute PSD using Welch's method
        # mne.time_frequency.psd_array_welch or evoked.compute_psd (mne >= 1.3)
        # We will use evoked.compute_psd which is the modern MNE API
        psd_spectrum = evoked.compute_psd(method='welch', picks=picks, fmin=0, fmax=50, verbose=False)
        psds, freqs = psd_spectrum.get_data(return_freqs=True)
        
        results = []
        for idx, ch_name in enumerate(ch_names):
            ch_psd = psds[idx]
            
            for band_name, (fmin, fmax) in bands.items():
                # Find indices corresponding to the frequency band
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                
                # Sum (or integrate) the power in the band
                band_power = np.sum(ch_psd[freq_mask])
                
                results.append({
                    'channel': ch_name,
                    'band_name': band_name,
                    'power': band_power
                })
                
        return pd.DataFrame(results)

    def get_peak_frequency(self, fmin: float = 1.0, fmax: float = 50.0, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Find the natural frequency (frequency with highest power) in a specific range.
        
        Parameters
        ----------
        fmin : float, default 1.0
            Minimum frequency of interest.
        fmax : float, default 50.0
            Maximum frequency of interest.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'peak_frequency' (Hz), and 'peak_power'.
        """
        evoked = self._extractor.evoked
        
        if channels is None:
            picks = mne.pick_types(evoked.info, eeg=True)
            ch_names = [evoked.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            picks = mne.pick_channels(evoked.ch_names, ch_names)
            
        psd_spectrum = evoked.compute_psd(method='welch', picks=picks, fmin=fmin, fmax=fmax, verbose=False)
        psds, freqs = psd_spectrum.get_data(return_freqs=True)
        
        results = []
        for idx, ch_name in enumerate(ch_names):
            ch_psd = psds[idx]
            peak_idx = np.argmax(ch_psd)
            
            results.append({
                'channel': ch_name,
                'peak_frequency': freqs[peak_idx],
                'peak_power': ch_psd[peak_idx]
            })
            
        return pd.DataFrame(results)
