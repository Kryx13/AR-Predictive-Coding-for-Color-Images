function [Rhat, Ghat, Bhat] = Predict_RGB(img, r, g, b)
% 输入 img 是原始图像，r/g/b 是预测系数
img = double(imread(img));
R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
[M,N] = size(R);

% 镜像边界扩展
Rp = padarray(R, [1 1], 'symmetric');
Gp = padarray(G, [1 1], 'symmetric');
Bp = padarray(B, [1 1], 'symmetric');

% 初始化预测矩阵
Rhat = zeros(M,N);
Ghat = zeros(M,N);
Bhat = zeros(M,N);

for i = 1:M
    for j = 1:N
        % 实际位置在 padding 后是 i+1,j+1
        x = i+1; y = j+1;

        % 提取相邻像素
        R_t = Rp(x-1,y);   R_l = Rp(x,y-1);     R_c = Rp(x,y);
        G_t = Gp(x-1,y);   G_l = Gp(x,y-1);     G_c = Gp(x,y);
        B_t = Bp(x-1,y);   B_l = Bp(x,y-1);

        % R 预测
        Rhat(i,j) = r(1)*R_t + r(2)*R_l + r(3)*G_t + r(4)*G_l + r(5)*B_t + r(6)*B_l;

        % G 预测
        Ghat(i,j) = g(1)*R_t + g(2)*R_l + g(3)*G_t + g(4)*G_l + g(5)*B_t + g(6)*B_l + g(7)*R_c;

        % B 预测
        Bhat(i,j) = b(1)*R_t + b(2)*R_l + b(3)*G_t + b(4)*G_l + b(5)*B_t + b(6)*B_l + b(7)*R_c + b(8)*G_c;
    end
end
RGBhat = cat(3, Rhat, Ghat, Bhat);      % 合并通道
RGBhat_clip = min(max(RGBhat, 0), 255); % 限制范围
imshow(uint8(RGBhat_clip));
title('Predicted Color Image');
end
