import numpy as np
import pandas as pd
import mne
from typing import List, Union, Dict, Any, Optional, Tuple

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

    def get_peak(self, peak_name: Optional[Union[str, List[str]]] = None, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Extract the peak amplitude and latency for a specific window.
        
        If the peak name starts with 'N' or 'n', it looks for the minimum value (negative peak).
        If it starts with 'P' or 'p', it looks for the maximum value (positive peak).
        Otherwise, it looks for the maximum absolute value.
        
        Parameters
        ----------
        peak_name : str, list of str, or None, default None
            The name of the peak to analyze (e.g., 'N15', 'P30'). Must be defined in the extractor.
            If None, all defined peaks are analyzed.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'peak_name', 'amplitude' (V), and 'latency' (s).
            
        Raises
        ------
        KeyError
            If a peak_name is not found in the extractor windows.
        """
        if peak_name is None:
            peak_names = list(self._extractor.windows.keys())
        elif isinstance(peak_name, str):
            peak_names = [peak_name]
        else:
            peak_names = peak_name

        evoked = self._extractor.evoked
        
        if channels is None:
            picks = mne.pick_types(evoked.info, eeg=True)
            ch_names = [evoked.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            
        results = []
        for p_name in peak_names:
            window = self._extractor[p_name]
            
            # Crop data to the specific window
            evoked_cropped = evoked.copy().crop(tmin=window[0], tmax=window[1])
            data = evoked_cropped.get_data(picks=ch_names)
            times = evoked_cropped.times
            
            is_negative = p_name.upper().startswith('N')
            is_positive = p_name.upper().startswith('P')
            
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
                    'peak_name': p_name,
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

    def get_area(self, peak_name: Optional[Union[str, List[str]]] = None, channels: Optional[Union[str, List[str]]] = None, absolute: bool = True) -> pd.DataFrame:
        """
        Calculate the Area Under the Curve (AUC) for a specific temporal window.
        
        Uses the trapezoidal rule to integrate the signal over time.
        
        Parameters
        ----------
        peak_name : str, list of str, or None, default None
            The name of the window/peak to analyze (e.g., 'N15', 'P30').
            If None, all defined peaks are analyzed.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
        absolute : bool, default False
            If True, calculates the area of the absolute signal (rectified).
            If False, calculates the net area (negative values subtract from the total).
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'window_name', and 'auc' (V*s).
        """
        if peak_name is None:
            peak_names = list(self._extractor.windows.keys())
        elif isinstance(peak_name, str):
            peak_names = [peak_name]
        else:
            peak_names = peak_name

        evoked = self._extractor.evoked
        
        if channels is None:
            picks = mne.pick_types(evoked.info, eeg=True)
            ch_names = [evoked.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            
        results = []
        for p_name in peak_names:
            window = self._extractor[p_name]
            evoked_cropped = evoked.copy().crop(tmin=window[0], tmax=window[1])
            data = evoked_cropped.get_data(picks=ch_names)
            times = evoked_cropped.times
            
            if absolute:
                data = np.abs(data)
                
            for idx, ch_name in enumerate(ch_names):
                ch_data = data[idx, :]
                
                # Trapezoidal integration
                auc = np.trapezoid(y=ch_data, x=times)
                
                results.append({
                    'channel': ch_name,
                    'window_name': p_name,
                    'auc': auc
                })
            
        return pd.DataFrame(results)

    def get_snr(self, peak_name: Optional[Union[str, List[str]]], baseline: Tuple[float, float], data: Optional[Union[mne.Evoked, Any]] = None, channels: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Calculate the Signal-to-Noise Ratio (SNR) for each channel.
        
        The SNR is calculated as the absolute peak amplitude within the window 
        divided by the standard deviation of the baseline period.
        
        Parameters
        ----------
        peak_name : str, list of str, or None
            The name of the window/peak to analyze (e.g., 'N15', 'P30').
            If None, all defined peaks are analyzed.
        baseline : tuple of float
            The baseline time window (start, end) in seconds (e.g., (-0.5, -0.01)).
        data : mne.Evoked, mne.Epochs, or None, default None
            The data to use for SNR calculation. If None, uses the Evoked 
            object stored in the extractor. If Epochs are provided, the baseline 
            standard deviation is calculated across all epochs and times, and the 
            peak is extracted from the averaged epochs.
        channels : str, list of str, or None, default None
            The channel(s) to analyze. If None, all EEG channels are used.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing 'channel', 'peak_name', 'peak_amplitude', 
            'baseline_std', and 'snr'.
        """
        if peak_name is None:
            peak_names = list(self._extractor.windows.keys())
        elif isinstance(peak_name, str):
            peak_names = [peak_name]
        else:
            peak_names = peak_name

        if data is None:
            data = self._extractor.evoked
            
        if channels is None:
            picks = mne.pick_types(data.info, eeg=True)
            ch_names = [data.ch_names[i] for i in picks]
        else:
            ch_names = [channels] if isinstance(channels, str) else channels
            
        # Extract baseline and calculate standard deviation
        data_baseline = data.copy().crop(tmin=baseline[0], tmax=baseline[1])
        baseline_arrays = data_baseline.get_data(picks=ch_names)
        
        if baseline_arrays.ndim == 3:
            # Epochs provided: baseline_arrays shape is (epochs, channels, times)
            baseline_std = np.std(baseline_arrays, axis=(0, 2))
            evoked_for_peak = data.copy().average()
        else:
            # Evoked provided: baseline_arrays shape is (channels, times)
            baseline_std = np.std(baseline_arrays, axis=1)
            evoked_for_peak = data
            
        results = []
        for p_name in peak_names:
            window = self._extractor[p_name]
            
            # Get peak amplitudes
            evoked_cropped = evoked_for_peak.copy().crop(tmin=window[0], tmax=window[1])
            peak_arrays = evoked_cropped.get_data(picks=ch_names)
            
            is_negative = p_name.upper().startswith('N')
            is_positive = p_name.upper().startswith('P')
            
            for idx, ch_name in enumerate(ch_names):
                ch_data = peak_arrays[idx, :]
                
                if is_negative:
                    peak_idx = np.argmin(ch_data)
                elif is_positive:
                    peak_idx = np.argmax(ch_data)
                else:
                    peak_idx = np.argmax(np.abs(ch_data))
                    
                peak_amp = ch_data[peak_idx]
                b_std = baseline_std[idx]
                
                if b_std == 0:
                    snr = np.nan
                else:
                    snr = np.abs(peak_amp) / b_std
                    
                results.append({
                    'channel': ch_name,
                    'peak_name': p_name,
                    'peak_amplitude': peak_amp,
                    'baseline_std': b_std,
                    'snr': snr
                })
            
        return pd.DataFrame(results)
