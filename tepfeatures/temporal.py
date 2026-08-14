import numpy as np
import pandas as pd
import mne
from typing import List, Union, Dict, Any, Optional

class TemporalFeatures:
    """
    Class for extracting temporal features from TMS-Evoked Potentials (TEPs).
    
    This class calculates peak amplitudes, latencies, and morphological features
    (e.g., peak-to-peak amplitude, slope) within specific time windows.
    
    Parameters
    ----------
    extractor : TEPExtractor
        The parent extractor instance containing the Evoked data and windows.
    """
    
    def __init__(self, extractor: Any):
        # We use Any for type hint to avoid circular imports, but it expects a TEPExtractor.
        self._extractor = extractor

    def get_peak(self, peak_name: str, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Extract the peak amplitude and latency for a specific window.
        
        If the peak name starts with 'N' or 'n', it looks for the minimum value (negative peak).
        If it starts with 'P' or 'p', it looks for the maximum value (positive peak).
        Otherwise, it looks for the maximum absolute value.
        
        Parameters
        ----------
        peak_name : str
            The name of the peak to analyze (e.g., 'N15', 'P30'). Must be defined in the extractor.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'peak_name', 'amplitude' (V), and 'latency' (s).
            
        Raises
        ------
        KeyError
            If the peak_name is not found in the extractor windows.
        """
        window = self._extractor[peak_name]
        evoked = self._extractor.evoked
        
        if channels is None:
            picks = mne.pick_types(evoked.info, eeg=True)
            ch_names = [evoked.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            
        # Crop data to the specific window
        evoked_cropped = evoked.copy().crop(tmin=window[0], tmax=window[1])
        data = evoked_cropped.get_data(picks=ch_names)
        times = evoked_cropped.times
        
        results = []
        is_negative = peak_name.upper().startswith('N')
        is_positive = peak_name.upper().startswith('P')
        
        for idx, ch_name in enumerate(ch_names):
            ch_data = data[idx, :]
            
            if is_negative:
                peak_idx = np.argmin(ch_data)
            elif is_positive:
                peak_idx = np.argmax(ch_data)
            else:
                peak_idx = np.argmax(np.abs(ch_data))
                
            peak_amp = ch_data[peak_idx]
            peak_lat = times[peak_idx]
            
            results.append({
                'channel': ch_name,
                'peak_name': peak_name,
                'amplitude': peak_amp,
                'latency': peak_lat
            })
            
        return pd.DataFrame(results)

    def get_morphology(self, peak1: str, peak2: str, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Calculate morphological descriptors between two peaks.
        
        Calculates:
        - Peak-to-peak amplitude (absolute difference in amplitudes)
        - Inter-peak interval (difference in latencies)
        - Slope (amplitude difference / latency difference)
        
        Parameters
        ----------
        peak1 : str
            The name of the first peak.
        peak2 : str
            The name of the second peak.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing morphological features per channel.
        """
        df_p1 = self.get_peak(peak1, channels).set_index('channel')
        df_p2 = self.get_peak(peak2, channels).set_index('channel')
        
        results = []
        for ch in df_p1.index:
            amp1, lat1 = df_p1.loc[ch, 'amplitude'], df_p1.loc[ch, 'latency']
            amp2, lat2 = df_p2.loc[ch, 'amplitude'], df_p2.loc[ch, 'latency']
            
            ptp_amp = abs(amp2 - amp1)
            inter_peak_interval = abs(lat2 - lat1)
            
            if inter_peak_interval > 0:
                slope = (amp2 - amp1) / (lat2 - lat1)
            else:
                slope = np.nan
                
            results.append({
                'channel': ch,
                'pair': f"{peak1}-{peak2}",
                'ptp_amplitude': ptp_amp,
                'inter_peak_interval': inter_peak_interval,
                'slope': slope
            })
            
        return pd.DataFrame(results)
