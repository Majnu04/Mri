import scipy.io
import h5py
import numpy as np
import os

def load_mat_file_test(mat_path):
    """Test function matching the app's loader"""
    try:
        print(f"Loading .mat file: {mat_path}")
        
        # Try loading with scipy.io first
        try:
            mat_data = scipy.io.loadmat(mat_path)
            print("Loaded with scipy.io (MATLAB v7.0 format)")
        except NotImplementedError:
            # Try h5py for MATLAB v7.3 files
            print("Trying h5py for MATLAB v7.3 format...")
            mat_data = {}
            with h5py.File(mat_path, 'r') as f:
                def extract_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data = obj[:]
                            # Handle different data types
                            if data.dtype == 'object' or len(data.shape) == 0:
                                return
                            mat_data[name] = data
                            print(f"Loaded HDF5 dataset: {name} with shape {data.shape}")
                        except Exception as e:
                            print(f"Skipping dataset {name}: {str(e)}")
                
                f.visititems(extract_data)
        
        return mat_data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Examine the structure of brain tumor .mat files
mat_file = 'brainTumorData/1.mat'

data = load_mat_file_test(mat_file)
if data:
    print("\nKeys in the .mat file:")
    for key, value in data.items():
        if not key.startswith('__'):
            print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
            if hasattr(value, 'dtype'):
                print(f"    dtype: {value.dtype}")
            if hasattr(value, 'min') and hasattr(value, 'max'):
                print(f"    range: {value.min():.3f} to {value.max():.3f}")
            print()
    
    print("\nFile size:", os.path.getsize(mat_file), "bytes")