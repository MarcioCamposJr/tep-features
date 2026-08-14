import numpy as np
import mne
from tepfeatures.core import TEPExtractor

def test_phase_1():
    print("Testing Phase 1...")
    
    # Create fake evoked data
    info = mne.create_info(ch_names=['Cz', 'Fz', 'Pz'], sfreq=1000, ch_types='eeg')
    data = np.random.randn(3, 1000) * 1e-6  # 3 channels, 1 second of data
    evoked = mne.EvokedArray(data, info, tmin=-0.2)
    
    # Initialize extractor
    extractor = TEPExtractor(evoked)
    print(f"Initialized: {extractor}")
    
    # Test getting a window
    n15_window = extractor['N15']
    print(f"N15 default window: {n15_window}")
    assert n15_window == [0.010, 0.020]
    
    # Test setting a window
    extractor['N15'] = (0.012, 0.018)
    print(f"N15 new window: {extractor['N15']}")
    assert extractor['N15'] == [0.012, 0.018]
    
    # Test validation
    try:
        extractor['P30'] = [0.050, 0.010]  # Invalid: start > end
    except ValueError as e:
        print(f"Validation works (start > end): {e}")
        
    try:
        extractor['N100'] = "invalid"  # Invalid type
    except TypeError as e:
        print(f"Validation works (wrong type): {e}")
        
    print("Phase 1 tests passed!\n")

if __name__ == "__main__":
    test_phase_1()
