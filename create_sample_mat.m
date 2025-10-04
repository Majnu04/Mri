% MATLAB script to create a sample brain MRI volume for testing
% This creates a simple synthetic brain volume with some anomalies

% Create a 3D volume (64x64x32 slices)
[X, Y, Z] = meshgrid(-32:31, -32:31, -16:15);

% Create brain-like structure (ellipsoid)
brain = ((X/25).^2 + (Y/25).^2 + (Z/12).^2) < 1;

% Add some intensity variation
volume = brain .* (0.5 + 0.3 * sin(X/10) .* cos(Y/10));

% Add some tumor-like bright spots
tumor1 = ((X-10).^2 + (Y-5).^2 + (Z-3).^2) < 25;
tumor2 = ((X+8).^2 + (Y+12).^2 + (Z+2).^2) < 16;

% Add some cyst-like dark spots
cyst1 = ((X-15).^2 + (Y+10).^2 + (Z-5).^2) < 20;
cyst2 = ((X+12).^2 + (Y-8).^2 + (Z+4).^2) < 15;

% Combine all structures
volume(tumor1) = 0.9;  % Bright tumors
volume(tumor2) = 0.8;
volume(cyst1) = 0.1;   % Dark cysts
volume(cyst2) = 0.05;

% Add noise
volume = volume + 0.05 * randn(size(volume));

% Ensure non-negative values
volume = max(volume, 0);

% Save as .mat file with common variable name
save('sample_brain_mri.mat', 'volume');

fprintf('Created sample_brain_mri.mat with volume shape: %dx%dx%d\n', size(volume));