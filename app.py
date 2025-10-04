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
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip metadata keys
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 1000:  # Reasonable size for MRI data
                    if data.ndim >= 2:  # Accept 2D or 3D data
                        volume_data = data
                        print(f"Found volume data in key '{key}' with shape: {data.shape}")
                        break
        
        if volume_data is None:
            # Try common keys specifically
            for key in possible_keys:
                if key in mat_data:
                    data = mat_data[key]
                    if isinstance(data, np.ndarray):
                        volume_data = data
                        print(f"Found volume data in key '{key}' with shape: {data.shape}")
                        break
        
        if volume_data is None:
            raise ValueError(f"No suitable volume data found in .mat file. Available keys: {list(mat_data.keys())}")
        
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
            return load_mat_file(mat_files[0])
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
    vol = exposure.rescale_intensity(volume, out_range=(0, 1))
    
    # Multi-threshold approach for different anomaly types
    thresh_otsu = filters.threshold_otsu(vol)
    
    # Tumor detection (bright regions)
    tumor_mask = vol > (thresh_otsu * 1.2)
    
    # Cyst/dark lesion detection (dark regions in brain tissue)
    brain_mask = vol > (thresh_otsu * 0.3)
    cyst_mask = (vol < (thresh_otsu * 0.7)) & brain_mask
    
    # Combine anomalies
    combined_mask = tumor_mask | cyst_mask
    
    # Clean up using morphological operations and size filtering
    combined_mask = ndimage.binary_opening(combined_mask, structure=np.ones((3,3,3)))
    combined_mask = ndimage.binary_closing(combined_mask, structure=np.ones((2,2,2)))
    
    # Label and filter by size
    labeled = measure.label(combined_mask)
    props = measure.regionprops(labeled)
    final_mask = np.zeros_like(vol, dtype=bool)
    
    for prop in props:
        # Keep regions between 100-50000 voxels (adjustable)
        if 100 < prop.area < 50000:
            final_mask[labeled == prop.label] = True
    
    return final_mask, tumor_mask, cyst_mask


def volume_to_mesh(volume, level=0.5):
    """Convert volume to mesh using marching cubes with error handling"""
    try:
        verts, faces, normals, values = measure.marching_cubes(volume, level=level)
        if len(verts) == 0:
            # Create a minimal mesh if no vertices found
            print(f"Warning: No vertices found at level {level}, creating empty mesh")
            verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh
    except Exception as e:
        print(f"Error in marching cubes: {str(e)}")
        # Return a minimal mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        return trimesh.Trimesh(vertices=verts, faces=faces)


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
        if current_volume is None:
            return jsonify({'error': 'No volume loaded', 'loaded': False}), 400
        
        return jsonify({
            'shape': current_volume.shape,
            'total_slices': current_volume.shape[0],
            'loaded': True,
            'data_type': str(current_volume.dtype),
            'data_range': [float(current_volume.min()), float(current_volume.max())]
        })
    except Exception as e:
        print(f"Error in volume_info: {str(e)}")
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
    progress_data = processing_progress.get(session_id, {
        'status': 'not_found', 
        'progress': 0, 
        'message': 'Session not found'
    })
    return jsonify(progress_data)


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
        
        # Store volume data
        current_volume = volume_data
        
        # Perform segmentation with progress updates
        processing_progress[session_id].update({
            'progress': 70,
            'message': 'Detecting tumors and anomalies...'
        })
        
        if HEAVY_PROCESSING_AVAILABLE:
            final_mask, tumor_mask, cyst_mask = enhanced_segmentation(volume_data)
            current_tumor_mask = tumor_mask
            current_cyst_mask = cyst_mask
        else:
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
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            dicom_count = 0
            
            for filename in file_list:
                if filename.lower().endswith('.dcm') or not os.path.splitext(filename)[1]:
                    dicom_count += 1
            
            if dicom_count == 0:
                raise ValueError("No DICOM files found in ZIP archive")
            
            processing_progress[session_id].update({
                'progress': 40,
                'message': f'Found {dicom_count} DICOM files. Loading...'
            })
            
            # Process DICOM files with progress updates
            processed_count = 0
            for filename in file_list:
                if filename.lower().endswith('.dcm') or not os.path.splitext(filename)[1]:
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

def process_large_file(file, session_id):
    """Process large DICOM files with progress tracking and memory optimization"""
    global current_volume, current_segmentation, current_tumor_mask, current_cyst_mask
    
    try:
        # Update progress: Starting
        processing_progress[session_id].update({
            'status': 'processing',
            'progress': 10,
            'message': 'Reading file...'
        })
        
        filename = file.filename.lower()
        
        if filename.endswith('.zip'):
            # Process ZIP file with chunking
            processing_progress[session_id].update({
                'progress': 20,
                'message': 'Extracting ZIP archive...'
            })
            
            volume_data = process_zip_chunked(file, session_id)
            
        elif filename.endswith('.mat'):
            # Process MATLAB file
            processing_progress[session_id].update({
                'progress': 20,
                'message': 'Loading MATLAB file...'
            })
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as temp_file:
                file.save(temp_file.name)
                volume_data = load_mat_file(temp_file.name)
                os.unlink(temp_file.name)
                
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        if volume_data is None:
            raise ValueError("Failed to extract volume data from file")
        
        # Update progress: Processing volume
        processing_progress[session_id].update({
            'progress': 50,
            'message': 'Processing 3D volume...'
        })
        
        # Store volume data
        current_volume = volume_data
        
        # Perform segmentation with progress updates
        processing_progress[session_id].update({
            'progress': 70,
            'message': 'Detecting tumors and anomalies...'
        })
        
        if HEAVY_PROCESSING_AVAILABLE:
            final_mask, tumor_mask, cyst_mask = enhanced_segmentation(volume_data)
            current_tumor_mask = tumor_mask
            current_cyst_mask = cyst_mask
        else:
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
        if HEAVY_PROCESSING_AVAILABLE:
            # Generate brain mesh
            brain_mesh = volume_to_mesh(volume)
            if brain_mesh:
                mesh_data['brain.obj'] = brain_mesh
            
            # Generate tumor mesh if available
            if tumor_mask is not None and np.any(tumor_mask):
                tumor_mesh = volume_to_mesh(tumor_mask.astype(float))
                if tumor_mesh:
                    mesh_data['tumors.obj'] = tumor_mesh
            
            # Generate cyst mesh if available
            if cyst_mask is not None and np.any(cyst_mask):
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
    """Generate a simple mesh representation when advanced processing is not available"""
    try:
        # Create a simple cube mesh as fallback
        vertices = [
            "v -1 -1 -1",
            "v 1 -1 -1", 
            "v 1 1 -1",
            "v -1 1 -1",
            "v -1 -1 1",
            "v 1 -1 1",
            "v 1 1 1",
            "v -1 1 1"
        ]
        
        faces = [
            "f 1 2 3 4",  # bottom
            "f 8 7 6 5",  # top
            "f 1 5 6 2",  # front
            "f 3 7 8 4",  # back
            "f 1 4 8 5",  # left
            "f 2 6 7 3"   # right
        ]
        
        return "\n".join(["# Simple mesh"] + vertices + faces)
        
    except Exception as e:
        return f"# Error generating mesh: {e}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"


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
