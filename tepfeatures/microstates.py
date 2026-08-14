import numpy as np
import pandas as pd
import mne
from typing import Any, Dict, Optional

class MicrostateFeatures:
    """
    Class for extracting dynamic features based on EEG Microstates from TEPs.
    
    This class provides the basic structure to identify periods with stable 
    topography in the averaged signal.
    
    Parameters
    ----------
    extractor : TEPExtractor
        The parent extractor instance containing the Evoked data and windows.
    """
    
    def __init__(self, extractor: Any):
        self._extractor = extractor
        self.microstates_: Optional[np.ndarray] = None
        self.segmentation_: Optional[np.ndarray] = None
        
    def fit(self, n_states: int = 4, max_iter: int = 100) -> 'MicrostateFeatures':
        """
        Fit a basic microstate clustering model on the evoked data.
        
        This is a simplified modified K-Means (ignoring polarity) to find 
        stable topographical patterns.
        
        Parameters
        ----------
        n_states : int, default 4
            The number of microstate classes to find.
        max_iter : int, default 100
            Maximum number of iterations for the clustering algorithm.
            
        Returns
        -------
        self
            Returns the instance itself.
        """
        evoked = self._extractor.evoked
        picks = mne.pick_types(evoked.info, eeg=True)
        data = evoked.get_data(picks=picks)
        
        # Calculate GFP
        gfp = np.std(data, axis=0)
        
        # Initialize cluster centers (templates) with data at random time points
        # where GFP is relatively high to avoid noise
        idx = np.argsort(gfp)[-n_states:]
        templates = data[:, idx]
        
        # Basic Modified K-Means (ignoring polarity)
        for _ in range(max_iter):
            # Calculate spatial correlation between all time points and templates
            # Dot product of normalized data and templates
            norm_data = data / (np.linalg.norm(data, axis=0) + 1e-10)
            norm_templates = templates / (np.linalg.norm(templates, axis=0)[:, None] + 1e-10).T
            
            # shape: (n_states, n_times)
            activation = np.dot(norm_templates.T, norm_data) 
            
            # Assign each time point to the template with max absolute correlation
            segmentation = np.argmax(np.abs(activation), axis=0)
            
            # Update templates
            new_templates = np.zeros_like(templates)
            for state in range(n_states):
                state_data = data[:, segmentation == state]
                if state_data.shape[1] > 0:
                    # The first principal component is a good estimator for the template
                    # but for MVP, we just take the mean of the absolute values or 
                    # simply the mean (simplified)
                    new_templates[:, state] = np.mean(state_data, axis=1)
            
            # Check convergence (simplified)
            if np.allclose(templates, new_templates, atol=1e-5):
                break
            templates = new_templates
            
        self.microstates_ = templates
        self.segmentation_ = segmentation
        return self
        
    def get_features(self) -> pd.DataFrame:
        """
        Extract dynamic features from the fitted microstates.
        
        Extracts:
        - duration: Total duration of the state (in seconds)
        - order: The first time this state appears
        - mean_gfp: Average GFP during this state
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the microstate features.
            
        Raises
        ------
        RuntimeError
            If the `fit` method hasn't been called yet.
        """
        if self.segmentation_ is None:
            raise RuntimeError("You must call `fit` before extracting features.")
            
        evoked = self._extractor.evoked
        picks = mne.pick_types(evoked.info, eeg=True)
        data = evoked.get_data(picks=picks)
        gfp = np.std(data, axis=0)
        times = evoked.times
        sfreq = evoked.info['sfreq']
        
        n_states = self.microstates_.shape[1]
        results = []
        
        for state in range(n_states):
            state_mask = (self.segmentation_ == state)
            
            if not np.any(state_mask):
                continue
                
            # Duration in seconds
            duration = np.sum(state_mask) / sfreq
            
            # Order (first onset)
            first_idx = np.argmax(state_mask) # argmax returns the first True
            order_time = times[first_idx]
            
            # Mean GFP
            mean_gfp = np.mean(gfp[state_mask])
            
            results.append({
                'microstate': f"MS_{state + 1}",
                'duration': duration,
                'onset_time': order_time,
                'mean_gfp': mean_gfp
            })
            
        # Sort by onset time to determine the temporal sequence
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('onset_time').reset_index(drop=True)
            df['temporal_order'] = df.index + 1
            
        return df
