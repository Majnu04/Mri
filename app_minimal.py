import os
import io
import zipfile
import base64
from flask import Flask, request, render_template, send_file, jsonify, url_for
import tempfile
import numpy as np

app = Flask(__name__)

# Environment-specific configuration
is_production = os.environ.get('VERCEL') is not None
if is_production:
    # Vercel production settings - minimal
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB
    app.config['UPLOAD_TIMEOUT'] = 30  # 30 seconds
else:
    # Local development settings
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB
    app.config['UPLOAD_TIMEOUT'] = 1800  # 30 minutes

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global storage for current session data
current_volume = None

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
            'Compress your files into a smaller ZIP archive',
            'Reduce the number of slices in your series',
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
        'message': 'MRI 3D Reconstruction service is running (minimal mode)' if is_production else 'MRI 3D Reconstruction service is running',
        'features': 'basic_upload' if is_production else 'full',
        'note': 'Upload functionality available - DICOM/MATLAB processing requires additional packages in local mode'
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Accept a ZIP file - minimal version for production"""
    global current_volume
    
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
                file_list = z.namelist()
                z.extractall(tmp)
                print(f"Extracted ZIP to {tmp}")
            
            # Analyze the uploaded files
            dcm_files = [f for f in file_list if f.lower().endswith('.dcm') or ('.' not in f and len(f) < 50)]
            mat_files = [f for f in file_list if f.lower().endswith('.mat')]
            
            if is_production:
                # Production mode - basic file analysis only
                return jsonify({
                    'success': True,
                    'message': 'File uploaded and analyzed successfully (production mode)',
                    'file_analysis': {
                        'total_files': len(file_list),
                        'dcm_files': len(dcm_files),
                        'mat_files': len(mat_files),
                        'file_size_mb': round(file_size / (1024*1024), 2)
                    },
                    'environment': 'production',
                    'note': 'For full processing, use local development mode with complete dependencies'
                })
            else:
                # Local development - would have full processing
                # This is just a placeholder for now
                return jsonify({
                    'success': True,
                    'message': 'File uploaded successfully (development mode)',
                    'file_analysis': {
                        'total_files': len(file_list),
                        'dcm_files': len(dcm_files),
                        'mat_files': len(mat_files),
                        'file_size_mb': round(file_size / (1024*1024), 2)
                    },
                    'environment': 'development',
                    'note': 'Install pydicom and h5py for full processing capabilities'
                })
                
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/volume_info')
def volume_info():
    """Return information about the loaded volume"""
    global current_volume
    
    if current_volume is None:
        return jsonify({
            'error': 'No volume loaded', 
            'loaded': False,
            'message': 'Upload a file first to load volume data'
        }), 400
    
    try:
        return jsonify({
            'loaded': True,
            'shape': current_volume.shape if current_volume is not None else None,
            'message': 'Volume data available'
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
        print("🏥 Starting MRI 3D Reconstruction Application (Minimal Mode)...")
        print("🌐 Server will be available at: http://localhost:5000")
        print("📊 Install pydicom and h5py for full functionality!")
        app.run(debug=True, host='0.0.0.0', port=5000)