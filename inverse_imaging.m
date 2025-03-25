% Load image and create mask
original = im2double(imread('cameraman.tif')); % 256x256 grayscale
[height, width] = size(original);
mask = rand(height, width) > 0.5;
corrupted = original .* mask;

% Convert to dlarray (SSCB format)
corrupted_dl = dlarray(corrupted, 'SSCB');
mask_dl = dlarray(mask, 'SSCB');

% Define U-Net architecture with proper upsampling
inputChannels = 32;
layers = [
    % Input: 32x32x32 noise
    imageInputLayer([32 32 inputChannels], 'Normalization', 'none')
    
    % Encoder (downsample to 8x8)
    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 2) % 32x32 -> 16x16
    reluLayer
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', 2) % 16x16 -> 8x8
    reluLayer
    
    % Decoder (upsample to 256x256)
    transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same') % 8x8 -> 16x16
    reluLayer
    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same')  % 16x16 -> 32x32
    reluLayer
    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same')  % 32x32 -> 64x64
    reluLayer
    transposedConv2dLayer(4, 16, 'Stride', 2, 'Cropping', 'same')  % 64x64 -> 128x128
    reluLayer
    transposedConv2dLayer(4, 8, 'Stride', 2, 'Cropping', 'same')   % 128x128 -> 256x256
    reluLayer
    
    % Final convolution to 1 channel
    convolution2dLayer(3, 1, 'Padding', 'same') % Output: 256x256x1
];
net = dlnetwork(layers); 

% Initialize fixed input noise (32x32x32)
z = dlarray(randn(32, 32, inputChannels, 1), 'SSCB');

% Training parameters
numIterations = 1000;
learningRate = 0.001;
averageGrad = [];
averageSqGrad = [];

% Training loop
figure;
for iter = 1:numIterations
    [loss, gradients] = dlfeval(@modelGradients, net, z, mask_dl, corrupted_dl);
    [net.Learnables, avgGrad, avgSqGrad] = adamupdate(net.Learnables, gradients, ...
        averageGrad, averageSqGrad, iter, learningRate);
    
    % Display progress
    if mod(iter, 100) == 0
        fprintf('Iteration %d, Loss: %.4f\n', iter, extractdata(loss));
        x_hat = predict(net, z);
        subplot(1,2,1);
        imshow(extractdata(x_hat), []);
        title(sprintf('Reconstruction (Iter %d)', iter));
        subplot(1,2,2);
        imshow(corrupted);
        title('Corrupted Image');
        drawnow;
    end
end

% Final output
final_reconstruction = extractdata(predict(net, z));
figure;
imshow(final_reconstruction, []);
title('Final Reconstructed Image');

% ========== Helper Function ========== %
function [loss, gradients] = modelGradients(net, z, mask, corrupted)
    [x_hat, state] = forward(net, z);
    net.State = state;
    x_hat_masked = x_hat .* mask; % Now x_hat is 256x256 (matches mask)
    loss = mean((x_hat_masked - corrupted).^2, 'all');
    gradients = dlgradient(loss, net.Learnables);
end