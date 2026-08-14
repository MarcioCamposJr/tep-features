import copy
from typing import Dict, List, Tuple, Union
import mne

from .config import DEFAULT_WINDOWS
from .temporal import TemporalFeatures

class TEPExtractor:
    """
    Main class for extracting features from TMS-Evoked Potentials (TEPs).
    
    This class acts as a facade, managing the temporal windows and providing
    access to temporal, spatial, and microstate features.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data (average over trials). Must be an instance of `mne.Evoked`.
        
    Raises
    ------
    TypeError
        If the provided `evoked` object is not an instance of `mne.Evoked`.
    """
    
    def __init__(self, evoked: mne.Evoked):
        if not isinstance(evoked, mne.Evoked):
            raise TypeError("The 'evoked' parameter must be an instance of mne.Evoked.")
            
        self.evoked = evoked
        self.windows: Dict[str, List[float]] = copy.deepcopy(DEFAULT_WINDOWS)
        
        # Sub-modules
        self.temporal = TemporalFeatures(self)
        self.spatial = None
        self.microstates = None
        
    def __getitem__(self, peak_name: str) -> List[float]:
        """
        Get the temporal window for a specific peak.
        
        Parameters
        ----------
        peak_name : str
            The name of the peak (e.g., 'N15', 'P30').
            
        Returns
        -------
        list of float
            The temporal window [start, end] in seconds.
            
        Raises
        ------
        KeyError
            If the peak_name is not defined in the windows.
        """
        if peak_name not in self.windows:
            raise KeyError(f"Peak '{peak_name}' is not defined. Available peaks: {list(self.windows.keys())}")
        return self.windows[peak_name]
        
    def __setitem__(self, peak_name: str, window: Union[List[float], Tuple[float, float]]):
        """
        Set or update the temporal window for a specific peak.
        
        Parameters
        ----------
        peak_name : str
            The name of the peak to add or update.
        window : list or tuple of two floats
            The new temporal window [start, end] in seconds.
            
        Raises
        ------
        ValueError
            If the window is not a sequence of exactly two floats, or if start >= end.
        TypeError
            If the window is not a list or tuple.
        """
        if not isinstance(window, (list, tuple)):
            raise TypeError("Window must be a list or tuple.")
            
        if len(window) != 2:
            raise ValueError("Window must contain exactly two values: [start, end].")
            
        try:
            start = float(window[0])
            end = float(window[1])
        except (ValueError, TypeError):
            raise ValueError("Window values must be numbers (floats).")
            
        if start >= end:
            raise ValueError("The start of the window must be strictly less than the end.")
            
        self.windows[peak_name] = [start, end]

    def __repr__(self) -> str:
        return f"<TEPExtractor | {len(self.windows)} windows defined, Evoked: {self.evoked.comment if self.evoked.comment else 'unnamed'}>"
