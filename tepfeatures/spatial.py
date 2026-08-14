import numpy as np
import pandas as pd
import mne
from typing import List, Union, Optional, Any, Tuple

class SpatialFeatures:
    """
    Class for extracting spatial and topographical features from TMS-Evoked Potentials (TEPs).
    
    This class calculates Global Mean Field Power (GFP), Local Mean Field Power (LMFP),
    and topographical similarities.
    
    Parameters
    ----------
    extractor : TEPExtractor
        The parent extractor instance containing the Evoked data and windows.
    """
    
    def __init__(self, extractor: Any):
        self._extractor = extractor

    def _calculate_field_power(self, data: np.ndarray) -> np.ndarray:
        """
        Helper method to calculate Mean Field Power (standard deviation across channels).
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_channels, n_times).
            
        Returns
        -------
        np.ndarray
            Field power time series of shape (n_times,).
        """
        # Typically GFP is the standard deviation across channels at each time point.
        # Alternatively, it could be the RMS. Using std is standard for EEG.
        return np.std(data, axis=0)

    def get_gfp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Global Mean Field Power (GFP) across all EEG channels.
        
        Returns
        -------
        times : np.ndarray
            Time vector.
        gfp : np.ndarray
            GFP time series.
        """
        evoked = self._extractor.evoked
        picks = mne.pick_types(evoked.info, eeg=True)
        data = evoked.get_data(picks=picks)
        
        gfp = self._calculate_field_power(data)
        return evoked.times, gfp

    def get_lmfp(self, channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Local Mean Field Power (LMFP) across a specific set of channels (ROI).
        
        Parameters
        ----------
        channels : list of str
            List of channel names defining the Region of Interest (ROI).
            
        Returns
        -------
        times : np.ndarray
            Time vector.
        lmfp : np.ndarray
            LMFP time series.
        """
        evoked = self._extractor.evoked
        data = evoked.get_data(picks=channels)
        
        lmfp = self._calculate_field_power(data)
        return evoked.times, lmfp

    def get_gfp_peak(self, peak_name: str) -> pd.DataFrame:
        """
        Extract the peak amplitude and latency of the GFP within a specific window.
        
        Parameters
        ----------
        peak_name : str
            The name of the peak to analyze (e.g., 'N15', 'P30').
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'peak_name', 'gfp_amplitude', and 'gfp_latency'.
        """
        window = self._extractor[peak_name]
        evoked_cropped = self._extractor.evoked.copy().crop(tmin=window[0], tmax=window[1])
        picks = mne.pick_types(evoked_cropped.info, eeg=True)
        data = evoked_cropped.get_data(picks=picks)
        
        gfp = self._calculate_field_power(data)
        times = evoked_cropped.times
        
        peak_idx = np.argmax(gfp)
        
        return pd.DataFrame([{
            'peak_name': peak_name,
            'gfp_amplitude': gfp[peak_idx],
            'gfp_latency': times[peak_idx]
        }])

    def topographical_similarity(self, peak1: str, peak2: str) -> pd.DataFrame:
        """
        Calculate the Cosine Similarity between the topographies at the GFP peaks
        of two different temporal windows.
        
        Parameters
        ----------
        peak1 : str
            The name of the first peak.
        peak2 : str
            The name of the second peak.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the 'pair' and the 'cosine_similarity'.
        """
        # Find GFP peak latencies to use as the representative time point for the topography
        df_p1 = self.get_gfp_peak(peak1)
        df_p2 = self.get_gfp_peak(peak2)
        
        lat1 = df_p1['gfp_latency'].iloc[0]
        lat2 = df_p2['gfp_latency'].iloc[0]
        
        evoked = self._extractor.evoked
        picks = mne.pick_types(evoked.info, eeg=True)
        
        # Get data at exact latencies
        idx1 = evoked.time_as_index(lat1)[0]
        idx2 = evoked.time_as_index(lat2)[0]
        
        topo1 = evoked.get_data(picks=picks)[:, idx1]
        topo2 = evoked.get_data(picks=picks)[:, idx2]
        
        # Cosine similarity calculation
        dot_product = np.dot(topo1, topo2)
        norm1 = np.linalg.norm(topo1)
        norm2 = np.linalg.norm(topo2)
        
        if norm1 == 0 or norm2 == 0:
            similarity = np.nan
        else:
            similarity = dot_product / (norm1 * norm2)
            
        return pd.DataFrame([{
            'pair': f"{peak1}-{peak2}",
            'cosine_similarity': similarity
        }])
