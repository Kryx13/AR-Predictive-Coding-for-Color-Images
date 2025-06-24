%preidction_rgB
function [Rhat, Ghat, Bhat] = Predict_RGB(img, r, g, b)
% Input img is the original image, r/g/b are prediction coefficients
img = double(imread(img));
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
RGBhat = cat(3, Rhat, Ghat, Bhat);      % Merge channels
RGBhat_clip = min(max(RGBhat, 0), 255); % Limit range
imshow(uint8(RGBhat_clip));
title('Predicted Color Image');
end
