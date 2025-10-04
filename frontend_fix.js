// Fixed frontend JavaScript for the optimized backend
// This should replace the upload handling in index.html

async function handleFileUpload(file) {
    if (!file) {
        updateStatus('Please select a ZIP file containing DICOM (.dcm) or MATLAB (.mat) series', 'error');
        return;
    }
    
    // Check file size
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    const maxSize = isLocal ? 10 * 1024 * 1024 * 1024 : 2 * 1024 * 1024 * 1024;
    const maxSizeLabel = isLocal ? '10GB' : '2GB';
    
    if (file.size > maxSize) {
        const environment = isLocal ? 'local development' : 'production (Vercel)';
        updateStatus(`File too large (${(file.size / (1024*1024)).toFixed(1)} MB). Maximum size for ${environment} is ${maxSizeLabel}.`, 'error');
        return;
    }
    
    updateStatus(`Starting upload of ${(file.size / (1024*1024)).toFixed(1)} MB file...`, 'processing');
    
    const fd = new FormData();
    fd.append('file', file);
    
    try {
        // Start upload with new progress tracking
        const resp = await fetch('/upload', { 
            method: 'POST', 
            body: fd
        });
        
        if (!resp.ok) {
            const errorData = await resp.json().catch(() => ({ message: 'Upload failed' }));
            updateStatus('Upload failed: ' + errorData.message, 'error');
            return;
        }
        
        const uploadResult = await resp.json();
        if (!uploadResult.success) {
            updateStatus('Upload failed: ' + (uploadResult.error || 'Unknown error'), 'error');
            return;
        }
        
        // Start progress tracking
        const sessionId = uploadResult.session_id;
        updateStatus('Upload successful! Processing medical imaging data...', 'processing');
        
        // Poll for progress updates
        await trackProgress(sessionId);
        
    } catch (error) {
        updateStatus('Error: ' + error.message, 'error');
    }
}

async function trackProgress(sessionId) {
    const maxAttempts = 300; // 5 minutes
    let attempts = 0;
    
    const checkProgress = async () => {
        try {
            const resp = await fetch(`/progress/${sessionId}`);
            const progress = await resp.json();
            
            if (progress.status === 'complete') {
                updateStatus('Processing complete! Loading 3D visualization...', 'success');
                
                // Load the processed data
                try {
                    const meshResp = await fetch('/get_meshes');
                    if (meshResp.ok) {
                        const blob = await meshResp.blob();
                        window.meshesBlobUrl = URL.createObjectURL(blob);
                        
                        // Set up download button
                        const downloadBtn = document.getElementById('download-btn');
                        if (downloadBtn) {
                            downloadBtn.onclick = () => { 
                                const a = document.createElement('a'); 
                                a.href = window.meshesBlobUrl; 
                                a.download = 'meshes.zip'; 
                                a.click(); 
                            };
                        }
                        
                        // Get volume info
                        const infoResp = await fetch('/volume_info');
                        if (infoResp.ok) {
                            window.volumeInfo = await infoResp.json();
                            
                            if (window.volumeInfo.loaded) {
                                setupSliceNavigation();
                                await loadAndRenderMeshesFromZip(await blob.arrayBuffer());
                                const mainContent = document.getElementById('main-content');
                                if (mainContent) mainContent.style.display = 'block';
                                updateStatus('Analysis complete! Explore your MRI data using the controls below.', 'success');
                            } else {
                                updateStatus('Volume data not loaded properly.', 'error');
                            }
                        } else {
                            updateStatus('Failed to get volume information.', 'error');
                        }
                    } else {
                        updateStatus('Processing completed but failed to load meshes.', 'error');
                    }
                } catch (error) {
                    updateStatus('Failed to load results: ' + error.message, 'error');
                }
                return;
            }
            
            if (progress.status === 'error') {
                updateStatus('Processing failed: ' + progress.message, 'error');
                return;
            }
            
            if (progress.status === 'processing' || progress.status === 'starting') {
                const progressPercentage = Math.max(0, Math.min(100, progress.progress || 0));
                updateStatus(`${progress.message || 'Processing...'} (${progressPercentage}%)`, 'processing');
                
                // Continue polling
                if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(checkProgress, 1000);
                } else {
                    updateStatus('Processing timeout. The file may be too large.', 'error');
                }
                return;
            }
            
        } catch (error) {
            updateStatus('Failed to track progress: ' + error.message, 'error');
        }
    };
    
    checkProgress();
}