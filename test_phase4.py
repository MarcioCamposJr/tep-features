import numpy as np
import mne
from tepfeatures.core import TEPExtractor

def test_phase_4():
    print("Testing Phase 4: Microstate Features...")
    
    # Create fake evoked data
    info = mne.create_info(ch_names=['Cz', 'Fz', 'Pz'], sfreq=1000, ch_types='eeg')
    data = np.random.randn(3, 1000) * 1e-6
    evoked = mne.EvokedArray(data, info, tmin=-0.2)
    
    # Initialize extractor
    extractor = TEPExtractor(evoked)
    
    # Test Microstates Fit
    print("\nFitting Microstates...")
    extractor.microstates.fit(n_states=4, max_iter=50)
    
    # Test Microstates Features extraction
    print("\nExtracting Microstate Features...")
    df_features = extractor.microstates.get_features()
    print(df_features)
    
    print("\nPhase 4 tests passed!\n")

if __name__ == "__main__":
    test_phase_4()
