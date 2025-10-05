import os
import io
import zipfile
import base64
from flask import Flask, request, render_template, send_file, jsonify, url_for
import tempfile
import numpy as np
import pydicom
import scipy.io
import h5py
import threading
import time
from collections import defaultdict

# Conditional imports for heavy dependencies
try:
    from skimage import exposure, filters, measure
    from scipy import ndimage
    import trimesh
    import meshio
    HEAVY_PROCESSING_AVAILABLE = True
except ImportError:
    HEAVY_PROCESSING_AVAILABLE = False
    # Lightweight fallbacks will be provided

app = Flask(__name__)

# Environment-specific configuration
is_production = os.environ.get('VERCEL') is not None
if is_production:
    # Vercel production settings
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
    app.config['UPLOAD_TIMEOUT'] = 30  # 30 seconds
else:
    # Local development settings
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB
    app.config['UPLOAD_TIMEOUT'] = 1800  # 30 minutes

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global storage for current session data
current_volume = None
current_segmentation = None
current_tumor_mask = None
current_cyst_mask = None

# Progress tracking for large file processing
processing_progress = defaultdict(lambda: {'status': 'idle', 'progress': 0, 'message': ''})
processing_results = {}

@app.errorhandler(413)
def request_entity_too_large(error):
    is_production = os.environ.get('VERCEL') is not None
    max_size_label = '2GB' if is_production else '10GB'
    environment = 'production (Vercel)' if is_production else 'local development'
    
    return jsonify({
        'error': 'File too large',
        'message': f'The uploaded file exceeds the maximum size limit of {max_size_label} for {environment}.',
        'max_size': max_size_label,
        'environment': environment,
        'suggestions': [
            'Compress your DICOM files into a smaller ZIP archive',
            'Reduce the number of slices in your series',
            'Use DICOM compression tools',
            'For files > 2GB: Run the application locally' if is_production else 'Split large files into smaller chunks'
        ]
    }), 413


def load_mat_file(mat_path):
    """Load MRI volume from .mat file with robust error handling for both v7.0 and v7.3 formats"""
    try:
        print(f"Loading .mat file: {mat_path}")
        
        # Try loading with scipy.io first (for older MATLAB files)
        try:
            mat_data = scipy.io.loadmat(mat_path)
            print("Loaded with scipy.io (MATLAB v7.0 format)")
        except NotImplementedError:
            # If that fails, try h5py for MATLAB v7.3 files
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
        
        # Common variable names in .mat files for MRI data
        possible_keys = ['volume', 'data', 'image', 'mri', 'brain', 'img', 'vol', 'cjdata', 'cjdata/image']
        
        # Find the volume data
        volume_data = None
        print(f"🔍 Available keys in mat_data: {list(mat_data.keys())}")
        
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip metadata keys
                data = mat_data[key]
                print(f"🔍 Checking key '{key}': type={type(data)}")
                if hasattr(data, 'shape'):
                    print(f"🔍 Key '{key}' shape: {data.shape}")
                if hasattr(data, 'size'):
                    print(f"🔍 Key '{key}' size: {data.size}")
                
                if isinstance(data, np.ndarray) and data.size > 1000:  # Reasonable size for MRI data
                    if data.ndim >= 2:  # Accept 2D or 3D data
                        volume_data = data
                        print(f"✅ Found volume data in key '{key}' with shape: {data.shape}")
                        break
                elif isinstance(data, dict):
                    print(f"🔍 Key '{key}' is a dictionary with sub-keys: {list(data.keys())}")
        
        if volume_data is None:
            # Try common keys specifically
            print("🔍 Trying common keys...")
            for key in possible_keys:
                if key in mat_data:
                    data = mat_data[key]
                    print(f"🔍 Common key '{key}': type={type(data)}")
                    if isinstance(data, np.ndarray):
                        volume_data = data
                        print(f"✅ Found volume data in common key '{key}' with shape: {data.shape}")
                        break
        
        if volume_data is None:
            raise ValueError(f"No suitable volume data found in .mat file. Available keys: {list(mat_data.keys())}")
        
        # Ensure we have a numpy array, not a dictionary or other type
        if not isinstance(volume_data, np.ndarray):
            if isinstance(volume_data, dict):
                # Try to extract from nested dictionary structure
                print(f"Volume data is a dictionary with keys: {list(volume_data.keys())}")
                found_array = False
                for sub_key in ['data', 'image', 'volume', 'img', 'cjdata', 'vol']:
                    if sub_key in volume_data:
                        potential_data = volume_data[sub_key]
                        if isinstance(potential_data, np.ndarray) and potential_data.size > 1000:
                            volume_data = potential_data
                            print(f"Extracted volume data from nested key '{sub_key}' with shape: {volume_data.shape}")
                            found_array = True
                            break
                
                if not found_array:
                    # Try the first numpy array we find
                    for key, value in volume_data.items():
                        if isinstance(value, np.ndarray) and value.size > 1000:
                            volume_data = value
                            print(f"Using volume data from nested key '{key}' with shape: {volume_data.shape}")
                            found_array = True
                            break
                
                if not found_array:
                    raise ValueError(f"No numpy array found in nested dictionary structure. Available data types: {[(k, type(v)) for k, v in volume_data.items()]}")
            else:
                # Try to convert to numpy array
                try:
                    volume_data = np.array(volume_data)
                    print(f"Converted data to numpy array with shape: {volume_data.shape}")
                except Exception as e:
                    raise ValueError(f"Cannot convert volume data to numpy array: {type(volume_data)}, error: {str(e)}")
        
        # Final verification that we have a proper numpy array
        if not isinstance(volume_data, np.ndarray):
            raise ValueError(f"Final volume_data is still not a numpy array: {type(volume_data)}")
        
        print(f"✅ Verified volume_data is numpy array with shape: {volume_data.shape}")
        
        # Ensure proper data type
        volume_data = volume_data.astype(np.float32)
        
        # Handle different data structures
        if volume_data.ndim == 2:
            # Single slice - create a volume with one slice
            volume_data = volume_data[np.newaxis, :, :]
            print(f"Converted 2D slice to 3D volume: {volume_data.shape}")
        elif volume_data.ndim == 3:
            # Check orientation - ensure slices are in first dimension
            min_dim = np.argmin(volume_data.shape)
            if min_dim != 0 and volume_data.shape[min_dim] < min(volume_data.shape) / 2:
                volume_data = np.moveaxis(volume_data, min_dim, 0)
                print(f"Reoriented volume to shape: {volume_data.shape}")
        elif volume_data.ndim == 4:
            # Handle 4D data (might have time dimension or multiple channels)
            if volume_data.shape[-1] <= 4:  # Likely channels (RGB, etc.)
                volume_data = np.mean(volume_data, axis=-1)  # Average channels
            else:
                volume_data = volume_data[:, :, :, 0]  # Take first volume
            print(f"Processed 4D data to shape: {volume_data.shape}")
        elif volume_data.ndim > 4:
            # Squeeze extra dimensions
            volume_data = np.squeeze(volume_data)
            print(f"Squeezed {volume_data.ndim}D data to shape: {volume_data.shape}")
        
        # Ensure we have a reasonable volume
        if volume_data.size < 1000:
            raise ValueError(f"Volume too small: {volume_data.shape}")
        
        print(f"Successfully loaded .mat file with final shape: {volume_data.shape}")
        print(f"Data range: {volume_data.min():.3f} to {volume_data.max():.3f}")
        return volume_data
        
    except Exception as e:
        print(f"Error loading .mat file {mat_path}: {str(e)}")
        raise


def load_volume_from_directory(directory):
    """Load volume from directory containing either DICOM or .mat files"""
    print(f"Scanning directory: {directory}")
    
    # Look for .mat files first
    mat_files = []
    dicom_files = []
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.lower().endswith('.mat'):
                mat_files.append(filepath)
            elif (filename.lower().endswith('.dcm') or 
                  (not filename.lower().endswith(('.zip', '.txt', '.md', '.json')) and '.' not in filename[-4:])):
                dicom_files.append(filepath)
    
    print(f"Found {len(mat_files)} .mat files and {len(dicom_files)} potential DICOM files")
    
    if mat_files:
        # Process .mat files
        if len(mat_files) == 1:
            print(f"Loading single .mat file: {mat_files[0]}")
            volume_result = load_mat_file(mat_files[0])
            print(f"Result from load_mat_file: type={type(volume_result)}")
            if hasattr(volume_result, 'shape'):
                print(f"Volume shape: {volume_result.shape}")
            else:
                print(f"⚠️ Result has no shape attribute!")
                if isinstance(volume_result, dict):
                    print(f"Result is a dictionary with keys: {list(volume_result.keys())}")
                    raise ValueError(f"load_mat_file returned a dictionary instead of numpy array")
            return volume_result
        else:
            # Multiple .mat files - try to load and stack them into a volume
            print(f"Processing {len(mat_files)} .mat files...")
            slices = []
            for mat_file in sorted(mat_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])):
                try:
                    vol = load_mat_file(mat_file)
                    if vol.ndim == 3:
                        # 3D volume - add all slices
                        for i in range(vol.shape[0]):
                            slices.append(vol[i])
                    elif vol.ndim == 2:
                        # 2D slice - add directly
                        slices.append(vol)
                    print(f"Added {vol.shape} from {os.path.basename(mat_file)}")
                except Exception as e:
                    print(f"Skipping {mat_file}: {str(e)}")
                    continue
            
            if len(slices) > 200:  # Limit to 200 slices to prevent memory issues
                print(f"Limiting to first 200 slices (out of {len(slices)}) to prevent memory issues")
                slices = slices[:200]
            
            if slices:
                print(f"Stacking {len(slices)} slices into volume...")
                stacked_volume = np.stack(slices, axis=0)
                print(f"Created volume with shape: {stacked_volume.shape}")
                return stacked_volume
            else:
                raise ValueError("No valid .mat files could be loaded")
    
    elif dicom_files:
        # Process DICOM files using existing function
        return load_dicom_series_from_dir(directory)
    
    else:
        raise ValueError("No .mat or DICOM files found in the uploaded data")


def load_dicom_series_from_dir(dicom_dir):
    """Load DICOM series from directory with robust error handling"""
    files = []
    # Recursively find all files
    for root, dirs, filenames in os.walk(dicom_dir):
        for filename in filenames:
            if not filename.startswith('.') and not filename.lower().endswith('.zip'):
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} potential files")
    
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            # Check if it has pixel data
            if hasattr(ds, 'pixel_array'):
                instance_num = getattr(ds, 'InstanceNumber', 0)
                slice_location = getattr(ds, 'SliceLocation', 0)
                # Use instance number or slice location for sorting
                sort_key = instance_num if instance_num != 0 else slice_location
                slices.append((sort_key, ds))
                print(f"Loaded DICOM: {os.path.basename(f)} - Instance: {instance_num}, Location: {slice_location}")
        except Exception as e:
            print(f"Skipping file {f}: {str(e)}")
            continue
    
    if not slices:
        raise ValueError(f"No valid DICOM files with pixel data found in {dicom_dir}")
    
    print(f"Successfully loaded {len(slices)} DICOM slices")
    
    # Sort slices
    slices.sort(key=lambda x: x[0])
    
    # Extract pixel arrays
    arrays = []
    for sort_key, ds in slices:
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
            arrays.append(pixel_array)
        except Exception as e:
            print(f"Error extracting pixel array: {str(e)}")
            continue
    
    if not arrays:
        raise ValueError("No pixel arrays could be extracted from DICOM files")
    
    # Check if all arrays have the same shape
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        print(f"Warning: Inconsistent slice shapes: {set(shapes)}")
        # Resize all to the most common shape
        from collections import Counter
        most_common_shape = Counter(shapes).most_common(1)[0][0]
        print(f"Resizing all slices to {most_common_shape}")
        
        resized_arrays = []
        for arr in arrays:
            if arr.shape != most_common_shape:
                from skimage.transform import resize
                resized = resize(arr, most_common_shape, preserve_range=True)
                resized_arrays.append(resized.astype(np.float32))
            else:
                resized_arrays.append(arr)
        arrays = resized_arrays
    
    volume = np.stack(arrays, axis=0)
    print(f"Created volume with shape: {volume.shape}")
    return volume

def dicom_to_volume(dicom_files):
    """Convert a list of DICOM files to a 3D volume array"""
    try:
        print(f"Converting {len(dicom_files)} DICOM files to volume...")
        
        if not dicom_files:
            raise ValueError("No DICOM files provided")
        
        # Sort DICOM files by slice location or instance number for proper ordering
        def get_slice_position(ds):
            # Try different DICOM tags to determine slice position
            if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
                return float(ds.SliceLocation)
            elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient is not None:
                return float(ds.ImagePositionPatient[2])  # Z coordinate
            elif hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
                return float(ds.InstanceNumber)
            else:
                return 0.0
        
        # Sort the DICOM files
        try:
            dicom_files.sort(key=get_slice_position)
        except Exception as e:
            print(f"Warning: Could not sort DICOM files by position: {e}")
            print("Using original order...")
        
        # Get pixel arrays from all DICOM files
        slices = []
        for i, ds in enumerate(dicom_files):
            try:
                if hasattr(ds, 'pixel_array'):
                    pixel_array = ds.pixel_array
                    
                    # Handle different data types and scaling
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope) if ds.RescaleSlope is not None else 1.0
                        intercept = float(ds.RescaleIntercept) if ds.RescaleIntercept is not None else 0.0
                        pixel_array = pixel_array * slope + intercept
                    
                    # Convert to float32 for processing
                    pixel_array = pixel_array.astype(np.float32)
                    slices.append(pixel_array)
                    
                    if i % 10 == 0:  # Progress update every 10 slices
                        print(f"Processed slice {i+1}/{len(dicom_files)}")
                        
            except Exception as e:
                print(f"Warning: Skipping slice {i}: {e}")
                continue
        
        if not slices:
            raise ValueError("No valid pixel data found in DICOM files")
        
        # Stack slices into 3D volume
        volume = np.stack(slices, axis=0)
        
        # Normalize the volume
        volume = volume.astype(np.float32)
        
        # Handle different intensity ranges
        volume_min = np.min(volume)
        volume_max = np.max(volume)
        
        if volume_max > volume_min:
            # Normalize to 0-1 range
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        print(f"Created volume with shape: {volume.shape}")
        print(f"Volume data type: {volume.dtype}")
        print(f"Volume range: [{np.min(volume):.3f}, {np.max(volume):.3f}]")
        
        return volume
        
    except Exception as e:
        print(f"Error converting DICOM to volume: {e}")
        raise


def enhanced_segmentation(volume):
    """Enhanced segmentation to detect tumors, cysts, and other anomalies"""
    
    # Type checking and debugging
    print(f"enhanced_segmentation input: type={type(volume)}")
    if hasattr(volume, 'shape'):
        print(f"Volume shape: {volume.shape}")
    else:
        print(f"⚠️ Volume has no shape attribute!")
        if isinstance(volume, dict):
            print(f"Volume is a dictionary with keys: {list(volume.keys())}")
            raise ValueError("enhanced_segmentation received a dictionary instead of numpy array")
        else:
            raise ValueError(f"enhanced_segmentation received unexpected type: {type(volume)}")
    
    # Ensure we have a numpy array
    if not isinstance(volume, np.ndarray):
        raise ValueError(f"Volume must be a numpy array, got {type(volume)}")
    
    try:
        print(f"Starting intensity rescaling for volume with shape: {volume.shape}")
        vol = exposure.rescale_intensity(volume, out_range=(0, 1))
        print(f"✅ Intensity rescaling successful, vol type: {type(vol)}, shape: {vol.shape}")
    except Exception as e:
        print(f"❌ Error during intensity rescaling: {str(e)}")
        raise
    
    try:
        print(f"Computing Otsu threshold...")
        # Multi-threshold approach for different anomaly types
        thresh_otsu = filters.threshold_otsu(vol)
        print(f"✅ Otsu threshold computed: {thresh_otsu}")
        
        # Tumor detection (bright regions)
        print(f"Detecting tumor regions...")
        tumor_mask = vol > (thresh_otsu * 1.2)
        print(f"✅ Tumor mask created, shape: {tumor_mask.shape}, positive voxels: {np.sum(tumor_mask)}")
        
        # Cyst/dark lesion detection (dark regions in brain tissue)
        print(f"Detecting brain and cyst regions...")
        brain_mask = vol > (thresh_otsu * 0.3)
        cyst_mask = (vol < (thresh_otsu * 0.7)) & brain_mask
        print(f"✅ Cyst mask created, shape: {cyst_mask.shape}, positive voxels: {np.sum(cyst_mask)}")
        
        # Combine anomalies
        print(f"Combining anomaly masks...")
        combined_mask = tumor_mask | cyst_mask
        print(f"✅ Combined mask created, shape: {combined_mask.shape}, positive voxels: {np.sum(combined_mask)}")
        
    except Exception as e:
        print(f"❌ Error during threshold computation or mask creation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        print(f"Applying morphological operations...")
        # Clean up using morphological operations and size filtering
        combined_mask = ndimage.binary_opening(combined_mask, structure=np.ones((3,3,3)))
        combined_mask = ndimage.binary_closing(combined_mask, structure=np.ones((2,2,2)))
        print(f"✅ Morphological operations completed")
        
        # Label and filter by size
        print(f"Labeling connected components...")
        labeled = measure.label(combined_mask)
        props = measure.regionprops(labeled)
        final_mask = np.zeros_like(vol, dtype=bool)
        print(f"✅ Found {len(props)} connected components")
        
        kept_regions = 0
        for prop in props:
            # Keep regions between 100-50000 voxels (adjustable)
            if 100 < prop.area < 50000:
                final_mask[labeled == prop.label] = True
                kept_regions += 1
        
        print(f"✅ Kept {kept_regions} regions after size filtering")
        print(f"✅ Segmentation completed successfully!")
        
        return final_mask, tumor_mask, cyst_mask
        
    except Exception as e:
        print(f"❌ Error during morphological operations or labeling: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def volume_to_mesh(volume, level=0.5):
    """Convert volume to mesh using marching cubes with error handling"""
    try:
        print(f"Running marching cubes on volume with shape {volume.shape}, level={level}")
        print(f"Volume data range: {volume.min():.3f} to {volume.max():.3f}")
        print(f"Volume data type: {volume.dtype}")
        
        # Normalize volume to 0-1 range for better threshold detection
        vol_normalized = exposure.rescale_intensity(volume, out_range=(0, 1))
        print(f"Normalized volume range: {vol_normalized.min():.3f} to {vol_normalized.max():.3f}")
        
        # Try different threshold levels if the first one fails
        thresholds = [level, 0.3, 0.7, 0.1, 0.9, 0.5]
        
        verts = None
        faces = None
        
        for threshold in thresholds:
            try:
                print(f"Attempting marching cubes with threshold {threshold}")
                verts, faces, normals, values = measure.marching_cubes(vol_normalized, level=threshold)
                if len(verts) > 0:
                    print(f"✅ Marching cubes successful with threshold {threshold}: {len(verts)} vertices, {len(faces)} faces")
                    break
                else:
                    print(f"⚠️ Marching cubes returned 0 vertices with threshold {threshold}")
            except Exception as e:
                print(f"❌ Marching cubes failed with threshold {threshold}: {e}")
                continue
        
        if verts is None or len(verts) == 0:
            # Create a more realistic fallback mesh
            print(f"⚠️ No vertices found, creating fallback brain-like mesh")
            # Create a simple brain-like ellipsoid mesh
            phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:20j]
            x = 50 * np.sin(phi) * np.cos(theta)
            y = 40 * np.sin(phi) * np.sin(theta)  
            z = 45 * np.cos(phi)
            
            # Create vertices and faces for a simple mesh
            vertices = []
            for i in range(len(phi)):
                for j in range(len(phi[0])):
                    vertices.append([x[i,j], y[i,j], z[i,j]])
            
            verts = np.array(vertices[:100])  # Limit to 100 vertices
            # Create simple triangular faces
            faces = []
            for i in range(0, len(verts)-2, 3):
                faces.append([i, i+1, i+2])
            faces = np.array(faces)
            
            print(f"Created fallback mesh: {len(verts)} vertices, {len(faces)} faces")
        
        # Create mesh using trimesh
        print(f"Creating trimesh with {len(verts)} vertices and {len(faces)} faces")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Verify mesh validity
        if mesh.is_watertight:
            print("✅ Mesh is watertight")
        else:
            print("⚠️ Mesh is not watertight")
        
        print(f"Mesh bounds: {mesh.bounds}")
        print(f"Mesh volume: {mesh.volume:.3f}")
        
        # Convert trimesh to OBJ format string
        obj_string = mesh.export(file_type='obj')
        if isinstance(obj_string, bytes):
            obj_string = obj_string.decode('utf-8')
        
        print(f"Generated OBJ string with {len(obj_string)} characters")
        print(f"OBJ preview: {obj_string[:200]}...")
        return obj_string
        
    except Exception as e:
        print(f"❌ Error in marching cubes: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal but visible OBJ mesh as string
        obj_fallback = """# Fallback brain mesh
v -25.0 -20.0 -22.5
v 25.0 -20.0 -22.5
v 0.0 20.0 -22.5
v 0.0 0.0 22.5
f 1 2 3
f 1 2 4
f 2 3 4
f 3 1 4
"""
        print(f"Using fallback OBJ with {len(obj_fallback)} characters")
        return obj_fallback


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/slice/<int:slice_num>')
def get_slice(slice_num):
    """Return slice data as base64 encoded image"""
    try:
        global current_volume, current_segmentation, current_tumor_mask, current_cyst_mask
        
        if current_volume is None:
            return jsonify({'error': 'No volume loaded'}), 400
        
        if slice_num < 0 or slice_num >= current_volume.shape[0]:
            return jsonify({'error': f'Slice number {slice_num} out of range (0-{current_volume.shape[0]-1})'}), 400
    
        # Get the slice
        slice_data = current_volume[slice_num]
        
        # Normalize to 0-255
        slice_norm = exposure.rescale_intensity(slice_data, out_range=(0, 255)).astype(np.uint8)
        
        # Convert to base64
        from PIL import Image
        img = Image.fromarray(slice_norm, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Get overlay data if available
        overlay_data = {}
        if current_segmentation is not None:
            try:
                seg_slice = current_segmentation[slice_num].astype(np.uint8) * 255
                seg_img = Image.fromarray(seg_slice, mode='L')
                seg_buffer = io.BytesIO()
                seg_img.save(seg_buffer, format='PNG')
                overlay_data['segmentation'] = base64.b64encode(seg_buffer.getvalue()).decode()
            except Exception as e:
                print(f"Error creating segmentation overlay: {e}")
        
        if current_tumor_mask is not None:
            try:
                tumor_slice = current_tumor_mask[slice_num].astype(np.uint8) * 255
                tumor_img = Image.fromarray(tumor_slice, mode='L')
                tumor_buffer = io.BytesIO()
                tumor_img.save(tumor_buffer, format='PNG')
                overlay_data['tumor'] = base64.b64encode(tumor_buffer.getvalue()).decode()
            except Exception as e:
                print(f"Error creating tumor overlay: {e}")
        
        if current_cyst_mask is not None:
            try:
                cyst_slice = current_cyst_mask[slice_num].astype(np.uint8) * 255
                cyst_img = Image.fromarray(cyst_slice, mode='L')
                cyst_buffer = io.BytesIO()
                cyst_img.save(cyst_buffer, format='PNG')
                overlay_data['cyst'] = base64.b64encode(cyst_buffer.getvalue()).decode()
            except Exception as e:
                print(f"Error creating cyst overlay: {e}")
        
        return jsonify({
            'slice': img_b64,
            'overlays': overlay_data,
            'slice_number': slice_num,
            'total_slices': current_volume.shape[0]
        })
    
    except Exception as e:
        print(f"Error in get_slice: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal error processing slice: {str(e)}'}), 500


@app.route('/volume_info')
def volume_info():
    """Return information about the loaded volume"""
    try:
        global current_volume
        print(f"🔍 volume_info called: current_volume is {current_volume is not None}")
        if current_volume is not None:
            print(f"🔍 current_volume shape: {current_volume.shape}")
        
        if current_volume is None:
            print("❌ volume_info: No volume loaded")
            return jsonify({'error': 'No volume loaded', 'loaded': False}), 400
        
        volume_info_data = {
            'shape': current_volume.shape,
            'total_slices': current_volume.shape[0],
            'loaded': True,
            'data_type': str(current_volume.dtype),
            'data_range': [float(current_volume.min()), float(current_volume.max())]
        }
        print(f"✅ volume_info returning: {volume_info_data}")
        return jsonify(volume_info_data)
    except Exception as e:
        print(f"❌ Error in volume_info: {str(e)}")
        return jsonify({'error': f'Internal error: {str(e)}', 'loaded': False}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for Vercel deployment"""
    is_production = os.environ.get('VERCEL') is not None
    return jsonify({
        'status': 'healthy',
        'environment': 'production' if is_production else 'development',
        'max_file_size': '2GB' if is_production else '10GB',
        'message': 'MRI 3D Reconstruction service is running'
    })

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get processing progress for a session"""
    print(f"Progress requested for session: {session_id}")
    print(f"Available sessions: {list(processing_progress.keys())}")
    
    progress_data = processing_progress.get(session_id, {
        'status': 'not_found', 
        'progress': 0, 
        'message': 'Session not found'
    })
    
    print(f"Returning progress: {progress_data}")
    return jsonify(progress_data)

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify server is responding"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'sessions': list(processing_progress.keys())
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Optimized upload endpoint with progress tracking and chunked processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate session ID for progress tracking
    session_id = str(int(time.time() * 1000))  # timestamp in milliseconds
    
    # Initialize progress tracking
    processing_progress[session_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Saving uploaded file...'
    }
    
    try:
        # Save file to temporary location first to avoid "closed file" issues
        filename = file.filename.lower()
        suffix = '.zip' if filename.endswith('.zip') else '.mat' if filename.endswith('.mat') else '.dcm'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Save the entire file content
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        processing_progress[session_id].update({
            'progress': 5,
            'message': 'File saved. Starting processing...'
        })
        
        # Start background processing with the saved file path
        def process_in_background():
            try:
                process_large_file_from_path(temp_file_path, filename, session_id)
            except Exception as e:
                processing_progress[session_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'Processing failed: {str(e)}'
                }
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except:
                    pass
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'File upload started. Use session_id to track progress.',
            'progress_url': f'/progress/{session_id}'
        })
        
    except Exception as e:
        processing_progress[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Upload failed: {str(e)}'
        }
        return jsonify({'error': str(e)}), 500

def process_matlab_from_zip(file_path, session_id):
    """Process MATLAB files from ZIP archive"""
    try:
        import tempfile
        import shutil
        from scipy.io import loadmat
        
        processing_progress[session_id].update({
            'progress': 50,
            'message': 'Extracting MATLAB files...'
        })
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract ZIP to temporary directory
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            processing_progress[session_id].update({
                'progress': 60,
                'message': 'Loading MATLAB data...'
            })
            
            # Process extracted files
            volume = load_volume_from_directory(temp_dir)
            
            processing_progress[session_id].update({
                'progress': 70,
                'message': 'Analyzing brain structure...'
            })
            
            # Generate segmentation and meshes
            segmentation, tumor_mask, cyst_mask = enhanced_segmentation(volume)
            
            processing_progress[session_id].update({
                'progress': 80,
                'message': 'Generating 3D models...'
            })
            
            # Generate meshes
            meshes = generate_meshes_for_volume(volume, tumor_mask, cyst_mask)
            
            processing_progress[session_id].update({
                'progress': 90,
                'message': 'Finalizing...'
            })
            
            # Store results globally
            global current_volume, current_segmentation, current_tumor_mask, current_cyst_mask
            current_volume = volume
            current_segmentation = segmentation  
            current_tumor_mask = tumor_mask
            current_cyst_mask = cyst_mask
            
            processing_progress[session_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'MATLAB processing completed successfully!',
                'volume_shape': volume.shape,
                'has_anomalies': np.any(tumor_mask) or np.any(cyst_mask)
            }
            
            return volume  # Return the volume data, not meshes
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        processing_progress[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'MATLAB processing failed: {str(e)}'
        }
        raise e

def process_large_file_from_path(file_path, filename, session_id):
    """Process large files from a saved file path with progress tracking and memory optimization"""
    global current_volume, current_segmentation, current_tumor_mask, current_cyst_mask
    
    try:
        # Update progress: Starting
        processing_progress[session_id].update({
            'status': 'processing',
            'progress': 10,
            'message': 'Reading file...'
        })
        
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.zip'):
            # Process ZIP file with chunking
            processing_progress[session_id].update({
                'progress': 20,
                'message': 'Extracting ZIP archive...'
            })
            
            volume_data = process_zip_chunked_from_path(file_path, session_id)
            
        elif filename_lower.endswith('.mat'):
            # Process MATLAB file
            processing_progress[session_id].update({
                'progress': 20,
                'message': 'Loading MATLAB file...'
            })
            
            volume_data = load_mat_file(file_path)
                
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        if volume_data is None:
            raise ValueError("Failed to extract volume data from file")
        
        # Update progress: Processing volume
        processing_progress[session_id].update({
            'progress': 50,
            'message': 'Processing 3D volume...'
        })
        
        # Ensure we have a proper numpy array before storing
        if not isinstance(volume_data, np.ndarray):
            print(f"⚠️ CRITICAL: volume_data is not numpy array: {type(volume_data)}")
            if isinstance(volume_data, dict):
                print(f"⚠️ volume_data is dict with keys: {list(volume_data.keys())}")
            raise ValueError(f"Volume data must be numpy array, got {type(volume_data)}")
        
        # Store volume data
        current_volume = volume_data.copy()  # Make a copy to prevent modification
        print(f"✅ STORED current_volume with type: {type(current_volume)}")
        print(f"✅ STORED current_volume with shape: {current_volume.shape}")
        print(f"✅ Global current_volume is now: {current_volume is not None}")
        
        # Perform segmentation with progress updates
        processing_progress[session_id].update({
            'progress': 70,
            'message': 'Detecting tumors and anomalies...'
        })
        
        if HEAVY_PROCESSING_AVAILABLE:
            print("🔍 Running enhanced segmentation...")
            # Use current_volume for segmentation to ensure consistency
            print(f"🔍 Passing current_volume with type: {type(current_volume)}")
            print(f"🔍 Passing current_volume with shape: {current_volume.shape}")
            
            # Double-check before passing to segmentation
            if not isinstance(current_volume, np.ndarray):
                print(f"⚠️ CRITICAL ERROR: current_volume corrupted!")
                raise ValueError(f"current_volume is not numpy array: {type(current_volume)}")
            
            final_mask, tumor_mask, cyst_mask = enhanced_segmentation(current_volume)
            current_tumor_mask = tumor_mask
            current_cyst_mask = cyst_mask
            print(f"✅ STORED tumor_mask: {current_tumor_mask is not None}")
            print(f"✅ STORED cyst_mask: {current_cyst_mask is not None}")
        else:
            print("⚠️ Heavy processing not available, skipping segmentation")
            current_tumor_mask = None
            current_cyst_mask = None
        
        # Final processing
        processing_progress[session_id].update({
            'progress': 90,
            'message': 'Finalizing results...'
        })
        
        # Store results
        processing_results[session_id] = {
            'volume_shape': volume_data.shape,
            'has_tumors': current_tumor_mask is not None and np.any(current_tumor_mask),
            'has_cysts': current_cyst_mask is not None and np.any(current_cyst_mask),
            'processing_complete': True
        }
        
        # Complete
        processing_progress[session_id].update({
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete! Ready for 3D visualization.'
        })
        
    except Exception as e:
        processing_progress[session_id].update({
            'status': 'error',
            'progress': 0,
            'message': f'Processing failed: {str(e)}'
        })
        raise

def process_zip_chunked_from_path(file_path, session_id):
    """Process ZIP files from a saved file path"""
    try:
        processing_progress[session_id].update({
            'progress': 30,
            'message': 'Analyzing ZIP contents...'
        })
        
        # Process ZIP file
        dicom_files = []
        mat_files = []
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Count different file types
            dicom_count = 0
            mat_count = 0
            
            for filename in file_list:
                if filename.startswith('__MACOSX/') or filename.startswith('.'):
                    continue  # Skip Mac metadata files
                    
                lower_filename = filename.lower()
                
                # Check for MATLAB files
                if lower_filename.endswith('.mat'):
                    mat_count += 1
                    continue
                
                # Check for DICOM files (more flexible detection)
                if (lower_filename.endswith('.dcm') or 
                    lower_filename.endswith('.dicom') or 
                    not os.path.splitext(filename)[1] or  # Files without extension
                    lower_filename.endswith('.ima') or    # Some DICOM variants
                    'dicom' in lower_filename):
                    
                    # Additional check: try to read as DICOM
                    try:
                        with zip_ref.open(filename) as dcm_file:
                            # Try to read just the header to verify it's DICOM
                            dcm_data = dcm_file.read(256)  # Read first 256 bytes
                            if b'DICM' in dcm_data or len(dcm_data) > 128:  # Basic DICOM check
                                dicom_count += 1
                    except:
                        continue
            
            print(f"Found {dicom_count} DICOM files and {mat_count} MATLAB files")
            
            # Check if we have any supported files
            if dicom_count == 0 and mat_count == 0:
                raise ValueError("No DICOM (.dcm) or MATLAB (.mat) files found in ZIP archive")
            
            if mat_count > 0:
                processing_progress[session_id].update({
                    'progress': 40,
                    'message': f'Found {mat_count} MATLAB files. Loading...'
                })
                # Process MATLAB files
                return process_matlab_from_zip(file_path, session_id)
            else:
                processing_progress[session_id].update({
                    'progress': 40,
                    'message': f'Found {dicom_count} DICOM files. Loading...'
                })
                # Process DICOM files
            
            # Process DICOM files with progress updates
            processed_count = 0
            for filename in file_list:
                if filename.startswith('__MACOSX/') or filename.startswith('.'):
                    continue
                    
                lower_filename = filename.lower()
                
                # Use the same flexible DICOM detection
                if (lower_filename.endswith('.dcm') or 
                    lower_filename.endswith('.dicom') or 
                    not os.path.splitext(filename)[1] or
                    lower_filename.endswith('.ima') or
                    'dicom' in lower_filename):
                    
                    try:
                        with zip_ref.open(filename) as dcm_file:
                            dcm_data = dcm_file.read()
                            ds = pydicom.dcmread(io.BytesIO(dcm_data))
                            if hasattr(ds, 'pixel_array'):
                                dicom_files.append(ds)
                        
                        processed_count += 1
                        progress = 40 + (processed_count / dicom_count) * 20  # 40-60% range
                        processing_progress[session_id].update({
                            'progress': int(progress),
                            'message': f'Loaded {processed_count}/{dicom_count} DICOM files...'
                        })
                        
                    except Exception as e:
                        print(f"Skipping invalid DICOM file {filename}: {str(e)}")
                        continue
        
        if not dicom_files:
            raise ValueError("No valid DICOM files could be loaded")
        
        # Convert to volume
        volume_data = dicom_to_volume(dicom_files)
        return volume_data
        
    except Exception as e:
        raise

# Note: process_large_file function removed - using process_large_file_from_path instead

def upload_old():
    """Accept a ZIP file of DICOMs and process them"""
    global current_volume, current_segmentation, current_tumor_mask, current_cyst_mask
    
    print("Upload request received")
    
    if 'dicom_zip' not in request.files:
        return jsonify({'error': 'No file provided', 'message': 'Please select a file to upload'}), 400
    
    f = request.files['dicom_zip']
    if f.filename == '':
        return jsonify({'error': 'No file selected', 'message': 'Please select a valid file'}), 400
    
    # Check file size before processing
    f.seek(0, 2)  # Seek to end
    file_size = f.tell()
    f.seek(0)  # Reset to beginning
    
    print(f"Processing file: {f.filename} (Size: {file_size / (1024*1024):.2f} MB)")
    
    # Check if file size exceeds limit based on environment
    is_production = os.environ.get('VERCEL') is not None
    max_size = 2 * 1024 * 1024 * 1024 if is_production else 10 * 1024 * 1024 * 1024  # 2GB prod, 10GB local
    max_size_label = '2GB' if is_production else '10GB'
    environment = 'production (Vercel)' if is_production else 'local development'
    
    if file_size > max_size:
        return jsonify({
            'error': 'File too large',
            'message': f'File size ({file_size / (1024*1024):.2f} MB) exceeds the maximum limit of {max_size_label} for {environment}',
            'file_size_mb': round(file_size / (1024*1024), 2),
            'max_size': max_size_label,
            'environment': environment
        }), 413
    
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, 'upload.zip')
            f.save(zpath)
            print(f"Saved upload to {zpath}")
            
            with zipfile.ZipFile(zpath, 'r') as z:
                z.extractall(tmp)
                print(f"Extracted ZIP to {tmp}")
            
            # find directory with medical imaging files
            data_dir = tmp
            files_found = False
            for root, dirs, files in os.walk(tmp):
                # Look for .mat files or .dcm files or files that might be DICOM
                medical_files = [f for f in files if (
                    f.lower().endswith('.mat') or 
                    f.lower().endswith('.dcm') or 
                    (not f.lower().endswith(('.zip', '.txt', '.md', '.json')) and '.' not in f[-4:])
                )]
                if medical_files:
                    data_dir = root
                    files_found = True
                    print(f"Found medical imaging directory: {data_dir} with {len(medical_files)} files")
                    break
            
            if not files_found:
                return 'No medical imaging files found in ZIP. Please ensure the ZIP contains .mat files or .dcm files.', 400
            
            # load series using unified loader
            print("Loading medical imaging data...")
            volume = load_volume_from_directory(data_dir)
            
            if volume.size == 0:
                return 'no medical imaging data found or volume is empty', 400
        
            # Store globally for slice serving
            current_volume = volume
            print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
            
            # Enhanced segmentation
            print("Performing segmentation...")
            seg, tumor_mask, cyst_mask = enhanced_segmentation(volume)
            current_segmentation = seg
            current_tumor_mask = tumor_mask
            current_cyst_mask = cyst_mask
            
            print(f"Segmentation complete. Found {np.sum(seg)} anomaly voxels")
            
            # build meshes
            print("Generating brain mesh...")
            brain_mesh = volume_to_mesh(volume / np.max(volume), level=0.2)
            
            print("Generating anomaly mesh...")
            anomaly_mesh = volume_to_mesh(seg.astype(np.float32), level=0.5)
            
            # Create separate meshes for different anomaly types if they exist
            tumor_mesh = None
            cyst_mesh = None
            
            if np.any(tumor_mask):
                tumor_mesh = volume_to_mesh(tumor_mask.astype(np.float32), level=0.5)
            
            if np.any(cyst_mask):
                cyst_mesh = volume_to_mesh(cyst_mask.astype(np.float32), level=0.5)
            
            # save meshes to temp files (within the temp directory context)
            brain_path = os.path.join(tmp, 'brain.obj')
            anomaly_path = os.path.join(tmp, 'anomalies.obj')
            
            brain_mesh.export(brain_path)
            anomaly_mesh.export(anomaly_path)
            
            # send back as files in zip
            out_zip = io.BytesIO()
            with zipfile.ZipFile(out_zip, 'w') as z:
                z.write(brain_path, 'brain.obj')
                z.write(anomaly_path, 'anomalies.obj')
                
                if tumor_mesh is not None:
                    tumor_path = os.path.join(tmp, 'tumors.obj')
                    tumor_mesh.export(tumor_path)
                    z.write(tumor_path, 'tumors.obj')
                
                if cyst_mesh is not None:
                    cyst_path = os.path.join(tmp, 'cysts.obj')
                    cyst_mesh.export(cyst_path)
                    z.write(cyst_path, 'cysts.obj')
            
            out_zip.seek(0)
            print("Processing complete, sending meshes")
            return send_file(out_zip, download_name='meshes.zip', as_attachment=True)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return f'Processing error: {str(e)}', 500

@app.route('/debug_globals')
def debug_globals():
    """Debug endpoint to check global variable states"""
    global current_volume, current_tumor_mask, current_cyst_mask
    
    return jsonify({
        'current_volume_loaded': current_volume is not None,
        'current_volume_shape': list(current_volume.shape) if current_volume is not None else None,
        'current_tumor_mask_loaded': current_tumor_mask is not None,
        'current_cyst_mask_loaded': current_cyst_mask is not None,
        'heavy_processing_available': HEAVY_PROCESSING_AVAILABLE
    })

@app.route('/test_mesh')
def test_mesh():
    """Test endpoint to generate a simple mesh for debugging"""
    try:
        # Create a simple test volume
        test_volume = np.random.rand(10, 10, 10)
        
        # Generate a simple mesh
        simple_mesh = generate_simple_mesh(test_volume)
        
        return jsonify({
            'success': True,
            'mesh_length': len(simple_mesh),
            'mesh_preview': simple_mesh[:200] + '...' if len(simple_mesh) > 200 else simple_mesh
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_meshes')
def get_meshes():
    """Generate and return mesh files for the current volume"""
    global current_volume, current_tumor_mask, current_cyst_mask
    
    if current_volume is None:
        return jsonify({'error': 'No volume loaded'}), 400
    
    try:
        # Generate meshes
        mesh_data = generate_meshes_for_volume(current_volume, current_tumor_mask, current_cyst_mask)
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in mesh_data.items():
                zip_file.writestr(filename, content)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='meshes.zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate meshes: {str(e)}'}), 500

def generate_meshes_for_volume(volume, tumor_mask, cyst_mask):
    """Generate mesh files for the volume and masks"""
    mesh_data = {}
    
    try:
        print(f"🔧 Generating meshes for volume shape: {volume.shape}")
        print(f"🔧 Volume data range: {volume.min():.3f} to {volume.max():.3f}")
        print(f"🔧 HEAVY_PROCESSING_AVAILABLE: {HEAVY_PROCESSING_AVAILABLE}")
        
        # Always generate brain mesh - this is the main visualization
        print("🧠 Generating brain mesh using marching cubes...")
        brain_mesh = volume_to_mesh(volume)
        if brain_mesh and len(brain_mesh) > 100:  # Check if we got a reasonable mesh
            mesh_data['brain.obj'] = brain_mesh
            print(f"✅ Brain mesh generated successfully, length: {len(brain_mesh)} characters")
        else:
            print("⚠️ Brain mesh generation failed or too small, using fallback")
            mesh_data['brain.obj'] = generate_simple_mesh(volume)
        
        if HEAVY_PROCESSING_AVAILABLE:
            # Generate tumor mesh if available
            if tumor_mask is not None and np.any(tumor_mask):
                print("🔴 Generating tumor mesh...")
                tumor_mesh = volume_to_mesh(tumor_mask.astype(float))
                if tumor_mesh and len(tumor_mesh) > 50:
                    mesh_data['tumors.obj'] = tumor_mesh
                    print(f"✅ Tumor mesh generated, length: {len(tumor_mesh)} characters")
                else:
                    print("⚠️ Tumor mesh generation failed")
            else:
                print("ℹ️ No tumor mask data available")
            
            # Generate cyst mesh if available
            if cyst_mask is not None and np.any(cyst_mask):
                print("🔵 Generating cyst mesh...")
                cyst_mesh = volume_to_mesh(cyst_mask.astype(float))
                if cyst_mesh and len(cyst_mesh) > 50:
                    mesh_data['cysts.obj'] = cyst_mesh
                    print(f"✅ Cyst mesh generated, length: {len(cyst_mesh)} characters")
                else:
                    print("⚠️ Cyst mesh generation failed")
            else:
                print("ℹ️ No cyst mask data available")
                cyst_mesh = volume_to_mesh(cyst_mask.astype(float))
                if cyst_mesh:
                    mesh_data['cysts.obj'] = cyst_mesh
        else:
            # Fallback: simple mesh generation
            mesh_data['brain.obj'] = generate_simple_mesh(volume)
            if tumor_mask is not None:
                mesh_data['tumors.obj'] = generate_simple_mesh(tumor_mask.astype(float))
            if cyst_mask is not None:
                mesh_data['cysts.obj'] = generate_simple_mesh(cyst_mask.astype(float))
    
    except Exception as e:
        print(f"Error generating meshes: {e}")
        # Provide basic fallback
        mesh_data['brain.obj'] = "# Basic brain mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"
    
    return mesh_data

def generate_simple_mesh(volume):
    """Generate a simple brain-like mesh representation when advanced processing is not available"""
    try:
        print("🔧 Generating simple brain-like mesh fallback...")
        
        # Create a brain-like ellipsoid mesh with multiple sections
        # Main brain body (ellipsoid)
        vertices = []
        faces = []
        
        # Create vertices for a brain-like shape
        import math
        
        # Brain stem (lower part)
        for i in range(8):
            angle = i * 2 * math.pi / 8
            x = 15 * math.cos(angle)
            y = 15 * math.sin(angle)
            z = -30
            vertices.append(f"v {x:.2f} {y:.2f} {z:.2f}")
        
        # Main brain (middle)  
        for i in range(12):
            angle = i * 2 * math.pi / 12
            for j in range(3):
                radius = 25 + j * 5
                x = radius * math.cos(angle)
                y = radius * math.sin(angle) * 0.8  # Slightly flattened
                z = -10 + j * 15
                vertices.append(f"v {x:.2f} {y:.2f} {z:.2f}")
        
        # Top of brain (cerebrum)
        for i in range(10):
            angle = i * 2 * math.pi / 10
            radius = 20
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) * 0.7
            z = 25
            vertices.append(f"v {x:.2f} {y:.2f} {z:.2f}")
        
        # Create faces to connect the vertices (simplified)
        # Bottom ring
        for i in range(8):
            next_i = (i + 1) % 8
            faces.append(f"f {i+1} {next_i+1} {i+9}")
        
        # Middle sections
        base = 9
        for layer in range(2):
            for i in range(12):
                next_i = (i + 1) % 12
                v1 = base + layer * 12 + i
                v2 = base + layer * 12 + next_i
                v3 = base + (layer + 1) * 12 + i
                v4 = base + (layer + 1) * 12 + next_i
                faces.append(f"f {v1} {v2} {v3}")
                faces.append(f"f {v2} {v4} {v3}")
        
        # Top cap
        top_base = 9 + 36  # After brain stem (8) + middle layers (36)
        center_top = len(vertices) + 1
        vertices.append("v 0.0 0.0 30.0")  # Top center point
        
        for i in range(10):
            next_i = (i + 1) % 10
            faces.append(f"f {top_base + i} {top_base + next_i} {center_top}")
        
        obj_content = "\n".join(["# Brain-like mesh fallback"] + vertices + faces)
        print(f"✅ Generated simple brain mesh with {len(vertices)} vertices and {len(faces)} faces")
        return obj_content
        
    except Exception as e:
        print(f"❌ Error generating simple mesh: {e}")
        # Ultra-simple fallback
        return """# Ultra-simple brain mesh
v -20.0 -15.0 -15.0
v 20.0 -15.0 -15.0
v 20.0 15.0 -15.0
v -20.0 15.0 -15.0
v -15.0 -10.0 15.0
v 15.0 -10.0 15.0
v 15.0 10.0 15.0
v -15.0 10.0 15.0
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
"""


import atexit

def cleanup_temp_files():
    """Clean up temporary mesh files on exit"""
    try:
        import os
        workspace_dir = os.getcwd()  # Use current working directory instead of __file__
        temp_dir = os.path.join(workspace_dir, 'temp_meshes')
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary mesh files")
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

# Register cleanup function
atexit.register(cleanup_temp_files)

# For Vercel deployment, we need to expose the app variable
# Vercel will automatically handle the WSGI server

if __name__ == '__main__':
    # Only run the development server when not on Vercel
    is_production = os.environ.get('VERCEL') is not None
    if not is_production:
        print("🏥 Starting 3D MRI Brain Tumor Visualization Application...")
        print("🌐 Server will be available at: http://localhost:5000")
        print("📊 Ready to process DICOM and MATLAB files!")
        app.run(debug=True, host='0.0.0.0', port=5000)
