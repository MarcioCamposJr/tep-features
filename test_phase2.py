import numpy as np
import mne
from tepfeatures.core import TEPExtractor

def test_phase_2():
    print("Testing Phase 2: Temporal Features...")
    
    # Create fake evoked data
    info = mne.create_info(ch_names=['Cz', 'Fz', 'Pz'], sfreq=1000, ch_types='eeg')
    # 1 second of data, 1000 points. Time goes from -0.2 to 0.8
    # N15 is at 0.010 - 0.020. Indices: 210 to 220
    # P30 is at 0.025 - 0.035. Indices: 225 to 235
    data = np.random.randn(3, 1000) * 1e-6
    evoked = mne.EvokedArray(data, info, tmin=-0.2)
    
    # Initialize extractor
    extractor = TEPExtractor(evoked)
    
    # Test get_peak
    print("Getting N15 peak for Cz...")
    df_n15 = extractor.temporal.get_peak('N15', channels='Cz')
    print(df_n15)
    
    print("Getting P30 peak for all channels...")
    df_p30 = extractor.temporal.get_peak('P30')
    print(df_p30)
    
    # Test get_morphology
    print("Getting N15-P30 morphology...")
    df_morph = extractor.temporal.get_morphology('N15', 'P30')
    print(df_morph)
    
    print("Phase 2 tests passed!\n")

if __name__ == "__main__":
    test_phase_2()
