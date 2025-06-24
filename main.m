%% main.m
% Project: Auto-Regressive Predictive Coding for RGB Images

% This script implements both global and local AR predictive coding
% strategies and compares their performance.

clear; close all; clc;

%load package
pkg load image

%% Configuration Parameters
config = struct();
config.image_path = 'images/tests/pic_tag.jpg';  % Change to your test image
config.block_size = 32;                      % Block size for local method
config.overlap = 0;                          % Block overlap (0 = no overlap)
config.save_results = true;                  % Save results to files
config.show_plots = true;                    % Display visualization plots

% Output directories
if ~exist('images/results/predicted', 'dir')
    mkdir('images/results/predicted');
end

fprintf('AR Predictive Coding for Color Images\n');
fprintf('=====================================\n\n');

%% Load and Prepare Image
fprintf('Loading image: %s\n', config.image_path);

try
    original_img = imread(config.image_path);
    if size(original_img, 3) ~= 3
        error('Image must be RGB color image');
    end
    original_img = double(original_img);
    [height, width, ~] = size(original_img);

    fprintf('Image loaded: %dx%d pixels\n', height, width);
catch ME
    error('Failed to load image: %s', ME.message);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GLOBAL METHOD - Single coefficient set for entire image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n GLOBAL METHOD\n');
fprintf('================\n');

tic;
fprintf('Calculating global AR coefficients...\n');

% Calculate global coefficients using Cal_para function
[r_global, g_global, b_global] = Cal_para(config.image_path);

fprintf('Global coefficients calculated (%.2f seconds)\n', toc);

% Display global coefficients
fprintf('\nGlobal AR Coefficients:\n');
fprintf('R: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', r_global);
fprintf('G: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', g_global);
fprintf('B: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', b_global);

tic;
fprintf('\n Performing global prediction...\n');

% Perform global prediction using Predict_RGB function
[R_pred_global, G_pred_global, B_pred_global] = Predict_RGB_Modified(original_img, r_global, g_global, b_global);

predicted_global = cat(3, R_pred_global, G_pred_global, B_pred_global);
predicted_global = min(max(predicted_global, 0), 255); % Clamp values

fprintf(' Global prediction completed (%.2f seconds)\n', toc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL METHOD - Block-based coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n LOCAL METHOD\n');
fprintf('===============\n');
fprintf(' Block size: %dx%d pixels\n', config.block_size, config.block_size);

tic;

% Initialize local prediction result
predicted_local = zeros(size(original_img));
coefficient_map = struct('r', [], 'g', [], 'b', []);

% Calculate number of blocks
num_blocks_h = ceil(height / config.block_size);
num_blocks_w = ceil(width / config.block_size);

fprintf(' Processing %d blocks (%dx%d grid)...\n', num_blocks_h * num_blocks_w, num_blocks_h, num_blocks_w);

% Store coefficients for each block
block_coeffs = cell(num_blocks_h, num_blocks_w);

for block_i = 1:num_blocks_h
    for block_j = 1:num_blocks_w
        % Calculate block boundaries
        row_start = (block_i - 1) * config.block_size + 1;
        row_end = min(block_i * config.block_size, height);
        col_start = (block_j - 1) * config.block_size + 1;
        col_end = min(block_j * config.block_size, width);

        % Extract block
        block_img = original_img(row_start:row_end, col_start:col_end, :);

        % Skip blocks that are too small (less than 8x8)
        if size(block_img, 1) < 8 || size(block_img, 2) < 8
            % Use global coefficients for small blocks
            r_local = r_global;
            g_local = g_global;
            b_local = b_global;
        else
            % Calculate local coefficients for this block
            try
                [r_local, g_local, b_local] = Cal_para_Block(block_img);
            catch
                % Fallback to global coefficients if calculation fails
                r_local = r_global;
                g_local = g_global;
                b_local = b_global;
            end
        end

        % Store coefficients
        block_coeffs{block_i, block_j} = struct('r', r_local, 'g', g_local, 'b', b_local);

        % Perform local prediction for this block
        [R_block, G_block, B_block] = Predict_RGB_Block(block_img, r_local, g_local, b_local);

        % Store results
        predicted_local(row_start:row_end, col_start:col_end, 1) = R_block;
        predicted_local(row_start:row_end, col_start:col_end, 2) = G_block;
        predicted_local(row_start:row_end, col_start:col_end, 3) = B_block;

        % Progress indicator
        if mod((block_i-1)*num_blocks_w + block_j, 10) == 0
            fprintf(' Processed %d/%d blocks\n', (block_i-1)*num_blocks_w + block_j, num_blocks_h * num_blocks_w);
        end
    end
end

predicted_local = min(max(predicted_local, 0), 255); % Clamp values

fprintf(' Local prediction completed (%.2f seconds)\n', toc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PERFORMANCE ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n PERFORMANCE ANALYSIS\n');
fprintf('=======================\n');

% Calculate errors
error_global = calculerMatriceErreur_Modified(original_img, predicted_global);
error_local = calculerMatriceErreur_Modified(original_img, predicted_local);

% Calculate entropy values
H_original = calc_entropie(uint8(original_img));
H_pred_global = calc_entropie(uint8(predicted_global));
H_pred_local = calc_entropie(uint8(predicted_local));
H_error_global = calc_entropie_Error(error_global);
H_error_local = calc_entropie_Error(error_local);

% Calculate MSE
mse_global = mean((original_img(:) - predicted_global(:)).^2);
mse_local = mean((original_img(:) - predicted_local(:)).^2);

% Calculate PSNR
psnr_global = 10 * log10(255^2 / mse_global);
psnr_local = 10 * log10(255^2 / mse_local);

% Display results
fprintf('\nEntropy Analysis:\n');
fprintf('Original image:      %.3f bits/pixel\n', H_original);
fprintf('Global prediction:   %.3f bits/pixel\n', H_pred_global);
fprintf('Local prediction:    %.3f bits/pixel\n', H_pred_local);
fprintf('Global error:        %.3f bits/pixel\n', H_error_global);
fprintf('Local error:         %.3f bits/pixel\n', H_error_local);

fprintf('\nQuality Metrics:\n');
fprintf('Global MSE:          %.2f\n', mse_global);
fprintf('Local MSE:           %.2f\n', mse_local);
fprintf('Global PSNR:         %.2f dB\n', psnr_global);
fprintf('Local PSNR:          %.2f dB\n', psnr_local);

fprintf('\nCompression Potential:\n');
fprintf('Global method:       %.1f%% entropy reduction\n', (1 - H_error_global/H_original) * 100);
fprintf('Local method:        %.1f%% entropy reduction\n', (1 - H_error_local/H_original) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if config.show_plots
    fprintf('\n VISUALIZATION\n');
    fprintf('================\n');

    % Main comparison figure
    figure('Name', 'AR Predictive Coding Results', 'Position', [100, 100, 1200, 800]);

    subplot(2, 3, 1);
    imshow(uint8(original_img));
    title('Original Image');

    subplot(2, 3, 2);
    imshow(uint8(predicted_global));
    title(sprintf('Global Prediction\nPSNR: %.2f dB', psnr_global));

    subplot(2, 3, 3);
    imshow(uint8(predicted_local));
    title(sprintf('Local Prediction\nPSNR: %.2f dB', psnr_local));

    subplot(2, 3, 4);
    error_img_global = uint8(abs(original_img - predicted_global) * 3); % Amplify for visibility
    imshow(error_img_global);
    title(sprintf('Global Error (×3)\nMSE: %.2f', mse_global));

    subplot(2, 3, 5);
    error_img_local = uint8(abs(original_img - predicted_local) * 3); % Amplify for visibility
    imshow(error_img_local);
    title(sprintf('Local Error (×3)\nMSE: %.2f', mse_local));

    subplot(2, 3, 6);
    comparison_data = [H_original, H_pred_global, H_pred_local, H_error_global, H_error_local];
    bar(comparison_data);
    set(gca, 'XTickLabel', {'Original', 'Global Pred', 'Local Pred', 'Global Err', 'Local Err'});
    ylabel('Entropy (bits/pixel)');
    title('Entropy Comparison');
    grid on;

    % Coefficient analysis figure for local method
    if num_blocks_h * num_blocks_w <= 100 % Only for reasonable number of blocks
        figure('Name', 'Local Coefficients Analysis', 'Position', [150, 150, 1000, 600]);

        % Extract first R coefficient for visualization
        r1_map = zeros(num_blocks_h, num_blocks_w);
        for i = 1:num_blocks_h
            for j = 1:num_blocks_w
                r1_map(i, j) = block_coeffs{i, j}.r(1);
            end
        end

        subplot(1, 2, 1);
        imagesc(r1_map);
        colorbar;
        title('R Channel - First Coefficient Map');
        xlabel('Block Column');
        ylabel('Block Row');

        subplot(1, 2, 2);
        hist(r1_map(:), 20);
        title('Distribution of R First Coefficient');
        xlabel('Coefficient Value');
        ylabel('Frequency');
        grid on;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SAVE RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if config.save_results
    fprintf('\n SAVING RESULTS\n');
    fprintf('=================\n');

    % Save predicted images
    imwrite(uint8(predicted_global), 'images/results/predicted_global.png');
    imwrite(uint8(predicted_local), 'images/results/predicted_local.png');
    imwrite(uint8(abs(original_img - predicted_global)), 'images/results/error_global.png');
    imwrite(uint8(abs(original_img - predicted_local)), 'images/results/error_local.png');

    % Save numerical results
    results = struct();
    results.config = config;
    results.entropy = struct('original', H_original, 'global_pred', H_pred_global, ...
                           'local_pred', H_pred_local, 'global_error', H_error_global, ...
                           'local_error', H_error_local);
    results.quality = struct('mse_global', mse_global, 'mse_local', mse_local, ...
                           'psnr_global', psnr_global, 'psnr_local', psnr_local);
    results.coefficients = struct('global', struct('r', r_global, 'g', g_global, 'b', b_global), ...
                                'local', block_coeffs);

    save('images/results/analysis_results.mat', 'results');

    fprintf('Results saved to images/results/\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SUMMARY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n SUMMARY\n');
fprintf('==========\n');
fprintf('Image: %dx%d pixels\n', height, width);
fprintf('Global method - PSNR: %.2f dB, Entropy reduction: %.1f%%\n', ...
        psnr_global, (1 - H_error_global/H_original) * 100);
fprintf('Local method  - PSNR: %.2f dB, Entropy reduction: %.1f%%\n', ...
        psnr_local, (1 - H_error_local/H_original) * 100);

if psnr_local > psnr_global
    fprintf(' Local method performs better (+%.2f dB PSNR)\n', psnr_local - psnr_global);
else
    fprintf(' Global method performs better (+%.2f dB PSNR)\n', psnr_global - psnr_local);
end

fprintf('\n sAnalysis complete!\n');


%% ========================================================================
%% AUXILIARY FUNCTIONS
%% ========================================================================

function [r, g, b] = Cal_para_Block(block_img)
    % Calculate AR coefficients for a single block
    % Similar to Cal_para.m but works with image data directly

    R = block_img(:,:,1);
    G = block_img(:,:,2);
    B = block_img(:,:,3);
    [M,N] = size(R);

    % Add boundary padding
    Rp = padarray(R, [1,1], 'symmetric');
    Gp = padarray(G, [1,1], 'symmetric');
    Bp = padarray(B, [1,1], 'symmetric');

    % Extract neighbor pixels
    R_left = Rp(2:M+1, 1:N);
    R_top = Rp(1:M,2:N+1);
    G_left = Gp(2:M+1, 1:N);
    G_top = Gp(1:M,2:N+1);
    B_left = Bp(2:M+1, 1:N);
    B_top = Bp(1:M,2:N+1);

    % Calculate covariance terms (same as Cal_para.m)
    RR00 = mean(mean(R.^2));
    RR01 = mean(mean(R.*R_top));
    RR10 = mean(mean(R.*R_left));
    RR11 = mean(mean(R_left .* R_top));

    GG00 = mean(mean(G.^2));
    GG01 = mean(mean(G.*G_top));
    GG10 = mean(mean(G.*G_left));
    GG11 = mean(mean(G_left .* G_top));

    BB00 = mean(mean(B.^2));
    BB01 = mean(mean(B.*B_top));
    BB10 = mean(mean(B.*B_left));
    BB11 = mean(mean(B_left .* B_top));

    RG00 = mean(mean(R.*G));
    RB00 = mean(mean(R.*B));
    GB00 = mean(mean(G.*B));

    RG01 = mean(mean(R.*G_top));
    RG10 = mean(mean(R.*G_left));
    RG11 = mean(mean(R_left.* G_top));

    RB01 = mean(mean(R.*B_top));
    RB10 = mean(mean(R.*B_left));
    RB11 = mean(mean(R_left.* B_top));

    GR01 = mean(mean(G.*R_top));
    GR10 = mean(mean(G.*R_left));
    GB01 = mean(mean(G.*B_top));
    GB10 = mean(mean(G.*B_left));
    GR11 = mean(mean(R_top.* G_left));
    GB11 = mean(mean(G_left.* B_top));

    BR01 = mean(mean(B.*R_top));
    BR10 = mean(mean(B.*R_left));
    BG01 = mean(mean(B.*G_top));
    BG10 = mean(mean(B.*G_left));
    BR11 = mean(mean(R_top.* B_left));
    BG11 = mean(mean(G_top.* B_left));

    % Build coefficient matrices (same as Cal_para.m)
    Kr = [RR00,RR11,RG00,RG11,RB00,RB11;
          RR11,RR00,GR11,RG00,BR11,RB00;
          RG00,GR11,GG00,GG11,GB00,GB11;
          RG11,RG00,GG11,GG00,BG11,GB00;
          RB00,BR11,GB00,BG11,BB00,BB11;
          RB11,RB00,GB11,GB00,BB11,BB00];

    Kg =[RR00,RR11,RG00,RG11,RB00,RB11,RR10;
         RR11,RR00,GR11,RG00,BR11,RB00,RR01;
         RG00,GR11,GG00,GG11,GB00,GB11,RG10;
         RG11,RG00,GG11,GG00,BG11,GB00,RG01;
         RB00,BR11,GB00,BG11,BB00,BB11,RB10;
         RB11,RB00,GB11,GB00,BB11,BB00,RB01;
         RR10,RR01,RG10,RG01,RB10,RB01,RR00];

    Kb =[RR00,RR11,RG00,RG11,RB00,RB11,RR10,GR10;
         RR11,RR00,GR11,RG00,BR11,RB00,RR01,GR01;
         RG00,GR11,GG00,GG11,GB00,GB11,RG10,RG10;
         RG11,RG00,GG11,GG00,BG11,GB00,RG01,GG01;
         RB00,BR11,GB00,BG11,BB00,BB11,RB10,GB10;
         RB11,RB00,GB11,GB00,BB11,BB00,RB01,GB01;
         RR10,RR01,RG10,RG01,RB10,RB01,RR00,RG00;
         GR10,GR01,GG10,GG01,GB10,GB01,RG00,GG00];

    Yr = [RR10,RR01,RG10,RG01,RB10,RB01]';
    Yg = [GR10,GR01,GG10,GG01,GB10,GB01,RG00]';
    Yb = [BR10,BR01,BG10,BG01,BB10,BB01,RB00,GB00]';

    % Solve for coefficients
    try
        r = Kr \ Yr;
        g = Kg \ Yg;
        b = Kb \ Yb;
    catch
        % If matrix is singular, use small regularization
        lambda = 1e-6 * trace(Kr) / size(Kr, 1);
        r = (Kr + lambda * eye(size(Kr))) \ Yr;
        g = (Kg + lambda * eye(size(Kg))) \ Yg;
        b = (Kb + lambda * eye(size(Kb))) \ Yb;
    end
end

function [Rhat, Ghat, Bhat] = Predict_RGB_Block(block_img, r, g, b)
    % Perform RGB prediction for a single block
    % Similar to Predict_RGB.m but works with image data directly

    R = block_img(:,:,1);
    G = block_img(:,:,2);
    B = block_img(:,:,3);
    [M,N] = size(R);

    % Mirror boundary extension
    Rp = padarray(R, [1 1], 'symmetric');
    Gp = padarray(G, [1 1], 'symmetric');
    Bp = padarray(B, [1 1], 'symmetric');

    % Initialize prediction matrices
    Rhat = zeros(M,N);
    Ghat = zeros(M,N);
    Bhat = zeros(M,N);

    for i = 1:M
        for j = 1:N
            % Actual position after padding is i+1,j+1
            x = i+1; y = j+1;

            % Extract neighboring pixels
            R_t = Rp(x-1,y);   R_l = Rp(x,y-1);     R_c = Rp(x,y);
            G_t = Gp(x-1,y);   G_l = Gp(x,y-1);     G_c = Gp(x,y);
            B_t = Bp(x-1,y);   B_l = Bp(x,y-1);

            % Predictions
            Rhat(i,j) = r(1)*R_t + r(2)*R_l + r(3)*G_t + r(4)*G_l + r(5)*B_t + r(6)*B_l;
            Ghat(i,j) = g(1)*R_t + g(2)*R_l + g(3)*G_t + g(4)*G_l + g(5)*B_t + g(6)*B_l + g(7)*R_c;
            Bhat(i,j) = b(1)*R_t + b(2)*R_l + b(3)*G_t + b(4)*G_l + b(5)*B_t + b(6)*B_l + b(7)*R_c + b(8)*G_c;
        end
    end
end

function [Rhat, Ghat, Bhat] = Predict_RGB_Modified(img, r, g, b)
    % Modified version of Predict_RGB that works with image data directly
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    [M,N] = size(R);

    % Mirror boundary extension
    Rp = padarray(R, [1 1], 'symmetric');
    Gp = padarray(G, [1 1], 'symmetric');
    Bp = padarray(B, [1 1], 'symmetric');

    % Initialize prediction matrix
    Rhat = zeros(M,N);
    Ghat = zeros(M,N);
    Bhat = zeros(M,N);

    for i = 1:M
        for j = 1:N
            % Actual position after padding is i+1,j+1
            x = i+1; y = j+1;

            % Extract neighboring pixels
            R_t = Rp(x-1,y);   R_l = Rp(x,y-1);     R_c = Rp(x,y);
            G_t = Gp(x-1,y);   G_l = Gp(x,y-1);     G_c = Gp(x,y);
            B_t = Bp(x-1,y);   B_l = Bp(x,y-1);

            % R prediction
            Rhat(i,j) = r(1)*R_t + r(2)*R_l + r(3)*G_t + r(4)*G_l + r(5)*B_t + r(6)*B_l;

            % G prediction
            Ghat(i,j) = g(1)*R_t + g(2)*R_l + g(3)*G_t + g(4)*G_l + g(5)*B_t + g(6)*B_l + g(7)*R_c;

            % B prediction
            Bhat(i,j) = b(1)*R_t + b(2)*R_l + b(3)*G_t + b(4)*G_l + b(5)*B_t + b(6)*B_l + b(7)*R_c + b(8)*G_c;
        end
    end
end
