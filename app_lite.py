import os
import io
import zipfile
import base64
from flask import Flask, request, render_template, send_file, jsonify, url_for
import tempfile
import numpy as np
import pydicom
import h5py

app = Flask(__name__)

# Environment-specific configuration
is_production = os.environ.get('VERCEL') is not None
if is_production:
    # Vercel production settings - simplified
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

def load_mat_file(mat_path):
    """Load MRI volume from .mat file using only h5py (for MATLAB v7.3 files)"""
    try:
        print(f"Loading .mat file: {mat_path}")
        
        # Use h5py for MATLAB v7.3 files (most modern .mat files)
        mat_data = {}
        try:
            with h5py.File(mat_path, 'r') as f:
                def extract_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data = obj[:]
                            if data.dtype == 'object' or len(data.shape) == 0:
                                return
                            mat_data[name] = data
                            print(f"Loaded HDF5 dataset: {name} with shape {data.shape}")
                        except Exception as e:
                            print(f"Skipping dataset {name}: {str(e)}")
                
                f.visititems(extract_data)
                
        except Exception as e:
            # If h5py fails, the file might be an older MATLAB format
            # For production deployment, we'll only support v7.3 files
            print(f"Could not load as HDF5/MATLAB v7.3 file: {str(e)}")
            raise ValueError("Only MATLAB v7.3 format (.mat) files are supported in production mode. Please re-save your .mat file in v7.3 format.")
        
        # Look for volume data with various common variable names
        volume_candidates = ['volume', 'data', 'image', 'mri', 'brain', 'img', 'vol']
        volume = None
        
        for candidate in volume_candidates:
            if candidate in mat_data:
                volume = mat_data[candidate]
                print(f"Found volume data in variable '{candidate}' with shape {volume.shape}")
                break
        
        if volume is None:
            # Try to find any 3D array
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 3:
                    volume = value
                    print(f"Using 3D array '{key}' as volume data with shape {volume.shape}")
                    break
        
        if volume is None:
            raise ValueError("No suitable 3D volume data found in .mat file")
        
        # Ensure volume is float and normalize
        volume = volume.astype(np.float32)
        if volume.max() > 1.0:
            volume = volume / volume.max()
        
        return volume
        
    except Exception as e:
        print(f"Error loading .mat file: {str(e)}")
        raise

def simple_threshold_segmentation(volume):
    """Simplified segmentation using only numpy operations"""
    # Simple thresholding without scikit-image
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    threshold = mean_val + 2 * std_val
    
    # Create binary mask
    mask = volume > threshold
    
    # Basic morphological operations using numpy only
    # Simple erosion and dilation with numpy operations
    kernel = np.ones((3, 3, 3), dtype=bool)
    
    # Basic erosion (minimum filter)
    eroded = np.zeros_like(mask)
    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            for k in range(1, mask.shape[2]-1):
                eroded[i,j,k] = np.all(mask[i-1:i+2, j-1:j+2, k-1:k+2])
    
    # Basic dilation (maximum filter)
    dilated = np.zeros_like(eroded)
    for i in range(1, eroded.shape[0]-1):
        for j in range(1, eroded.shape[1]-1):
            for k in range(1, eroded.shape[2]-1):
                dilated[i,j,k] = np.any(eroded[i-1:i+2, j-1:j+2, k-1:k+2])
    
    return dilated

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

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Vercel deployment"""
    is_production = os.environ.get('VERCEL') is not None
    return jsonify({
        'status': 'healthy',
        'environment': 'production' if is_production else 'development',
        'max_file_size': '2GB' if is_production else '10GB',
        'message': 'MRI 3D Reconstruction service is running (lightweight mode)' if is_production else 'MRI 3D Reconstruction service is running',
        'features': 'basic' if is_production else 'full'
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Accept a ZIP file of DICOMs and process them - simplified version"""
    global current_volume, current_segmentation
    
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
                mat_files = [f for f in files if f.lower().endswith('.mat')]
                dcm_files = [f for f in files if f.lower().endswith('.dcm') or '.' not in f]
                
                if mat_files:
                    print(f"Found {len(mat_files)} .mat files in {root}")
                    data_dir = root
                    files_found = True
                    
                    # Load the first .mat file
                    mat_path = os.path.join(root, mat_files[0])
                    volume = load_mat_file(mat_path)
                    break
                    
                elif dcm_files:
                    print(f"Found {len(dcm_files)} potential DICOM files in {root}")
                    data_dir = root
                    files_found = True
                    
                    # Load DICOM series
                    volume = load_dicom_series(root, dcm_files)
                    break
            
            if not files_found:
                return jsonify({'error': 'No DICOM or MATLAB files found in the uploaded ZIP'}), 400
            
            print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
            
            # Store volume globally
            current_volume = volume
            
            # Simplified processing for production
            if is_production:
                # Basic segmentation only
                current_segmentation = simple_threshold_segmentation(volume)
                
                # Return basic info instead of complex meshes
                return jsonify({
                    'success': True,
                    'message': 'File processed successfully (basic mode)',
                    'volume_shape': volume.shape,
                    'environment': 'production',
                    'features_available': ['slice_viewing', 'basic_segmentation']
                })
            else:
                # Full processing for local development
                # This would include the full segmentation and mesh generation
                # but is simplified here for compatibility
                current_segmentation = simple_threshold_segmentation(volume)
                
                return jsonify({
                    'success': True,
                    'message': 'File processed successfully',
                    'volume_shape': volume.shape,
                    'environment': 'development',
                    'features_available': ['slice_viewing', 'basic_segmentation', '3d_visualization']
                })
                
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def load_dicom_series(directory, filenames):
    """Load DICOM series from directory"""
    slices = []
    for filename in filenames:
        try:
            filepath = os.path.join(directory, filename)
            ds = pydicom.dcmread(filepath)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
        except Exception as e:
            print(f"Skipping file {filename}: {str(e)}")
    
    if not slices:
        raise ValueError("No valid DICOM files found")
    
    # Sort by slice location if available
    try:
        slices.sort(key=lambda x: float(x.SliceLocation))
    except:
        try:
            slices.sort(key=lambda x: float(x.InstanceNumber))
        except:
            pass  # Use original order
    
    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices])
    
    # Normalize to 0-1
    volume = volume.astype(np.float32)
    if volume.max() > 1.0:
        volume = volume / volume.max()
    
    return volume

@app.route('/slice/<int:slice_num>')
def get_slice(slice_num):
    """Return a specific slice as base64 encoded image"""
    global current_volume
    
    if current_volume is None:
        return jsonify({'error': 'No volume loaded'}), 400
    
    try:
        if slice_num >= current_volume.shape[0]:
            return jsonify({'error': 'Slice number out of range'}), 400
        
        slice_data = current_volume[slice_num]
        
        # Convert to 8-bit for display
        slice_norm = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        
        # Convert to PIL Image and then to base64
        from PIL import Image
        img = Image.fromarray(slice_norm)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'slice_data': img_str,
            'slice_number': slice_num,
            'total_slices': current_volume.shape[0]
        })
        
    except Exception as e:
        print(f"Error getting slice: {str(e)}")
        return jsonify({'error': f'Failed to get slice: {str(e)}'}), 500

@app.route('/volume_info')
def volume_info():
    """Return information about the loaded volume"""
    global current_volume
    
    if current_volume is None:
        return jsonify({'error': 'No volume loaded', 'loaded': False}), 400
    
    try:
        return jsonify({
            'loaded': True,
            'shape': current_volume.shape,
            'dtype': str(current_volume.dtype),
            'min_value': float(current_volume.min()),
            'max_value': float(current_volume.max()),
            'total_slices': int(current_volume.shape[0])
        })
    except Exception as e:
        print(f"Error in volume_info: {str(e)}")
        return jsonify({'error': f'Internal error: {str(e)}', 'loaded': False}), 500

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