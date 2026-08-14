import numpy as np
import mne
from tepfeatures.core import TEPExtractor

def test_phase_3():
    print("Testing Phase 3: Spatial Features...")
    
    # Create fake evoked data
    info = mne.create_info(ch_names=['Cz', 'Fz', 'Pz'], sfreq=1000, ch_types='eeg')
    data = np.random.randn(3, 1000) * 1e-6
    evoked = mne.EvokedArray(data, info, tmin=-0.2)
    
    # Initialize extractor
    extractor = TEPExtractor(evoked)
    
    # Test GFP
    print("\nGetting GFP...")
    times, gfp = extractor.spatial.get_gfp()
    print(f"GFP shape: {gfp.shape}, max value: {np.max(gfp)}")
    
    # Test LMFP
    print("\nGetting LMFP for ['Cz', 'Fz']...")
    times_lmfp, lmfp = extractor.spatial.get_lmfp(['Cz', 'Fz'])
    print(f"LMFP shape: {lmfp.shape}, max value: {np.max(lmfp)}")
    
    # Test GFP peak
    print("\nGetting GFP Peak for N15...")
    df_gfp_peak = extractor.spatial.get_gfp_peak('N15')
    print(df_gfp_peak)
    
    # Test Topographical Similarity
    print("\nGetting Topographical Similarity between N15 and P30...")
    df_topo_sim = extractor.spatial.topographical_similarity('N15', 'P30')
    print(df_topo_sim)
    
    print("\nPhase 3 tests passed!\n")

if __name__ == "__main__":
    test_phase_3()
