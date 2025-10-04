#!/usr/bin/env python3
"""
Create a sample .mat file with synthetic brain MRI data for testing
"""

import numpy as np
import scipy.io
import zipfile
import os

def create_sample_brain_volume():
    """Create a synthetic brain MRI volume with anomalies"""
    # Create coordinate grids
    x = np.linspace(-32, 31, 64)
    y = np.linspace(-32, 31, 64) 
    z = np.linspace(-16, 15, 32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create brain-like structure (ellipsoid)
    brain_mask = ((X/25)**2 + (Y/25)**2 + (Z/12)**2) < 1
    
    # Add some intensity variation
    volume = brain_mask * (0.5 + 0.3 * np.sin(X/10) * np.cos(Y/10))
    
    # Add tumor-like bright spots
    tumor1 = ((X-10)**2 + (Y-5)**2 + (Z-3)**2) < 25
    tumor2 = ((X+8)**2 + (Y+12)**2 + (Z+2)**2) < 16
    
    # Add cyst-like dark spots  
    cyst1 = ((X-15)**2 + (Y+10)**2 + (Z-5)**2) < 20
    cyst2 = ((X+12)**2 + (Y-8)**2 + (Z+4)**2) < 15
    
    # Combine all structures
    volume[tumor1] = 0.9  # Bright tumors
    volume[tumor2] = 0.8
    volume[cyst1] = 0.1   # Dark cysts
    volume[cyst2] = 0.05
    
    # Add some noise
    volume = volume + 0.05 * np.random.randn(*volume.shape)
    
    # Ensure non-negative values
    volume = np.maximum(volume, 0)
    
    return volume.astype(np.float32)

def main():
    print("Creating synthetic brain MRI data...")
    
    # Create the volume
    brain_volume = create_sample_brain_volume()
    print(f"Created volume with shape: {brain_volume.shape}")
    
    # Save as .mat file
    mat_filename = 'sample_brain_mri.mat'
    scipy.io.savemat(mat_filename, {
        'volume': brain_volume,
        'description': 'Synthetic brain MRI with tumors and cysts',
        'dimensions': brain_volume.shape
    })
    print(f"Saved {mat_filename}")
    
    # Create a ZIP file for easy upload
    zip_filename = 'sample_brain_data.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        zf.write(mat_filename)
    
    print(f"Created {zip_filename} for upload testing")
    print(f"File size: {os.path.getsize(zip_filename)} bytes")
    
    # Clean up the .mat file (keep only the zip)
    os.remove(mat_filename)
    
    print("\nTo test the application:")
    print(f"1. Open http://127.0.0.1:5000 in your browser")
    print(f"2. Upload the {zip_filename} file")
    print(f"3. The volume contains synthetic tumors and cysts for demonstration")

if __name__ == "__main__":
    main()