function [err_r, err_g, err_b, Rmed, Gmed, Bmed] = predictionRGB_nocenter(filename, r, g, b, delta)
    % 读取图像
    img = imread(filename);
    img = double(img);  
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    [M,N] = size(R);

    % calculate the mean value
Rmed = mean(R(:));
Gmed = mean(G(:));
Bmed = mean(B(:));
Rz = R - Rmed; Gz = G - Gmed; Bz = B - Bmed;

% initial error matrix
err_r = zeros(M,N);
err_g = zeros(M,N);
err_b = zeros(M,N);

% initial prediction
Rrec = zeros(M,N);
Grec = zeros(M,N);
Brec = zeros(M,N);

    for i = 1:M
        for j = 1:N
            % 邻域像素（已重建值）
            Rl = Rrec(max(i-1,1), j);
            Rt = Rrec(i, max(j-1,1));
            Gl = Grec(max(i-1,1), j);
            Gt = Grec(i, max(j-1,1));
            Bl = Brec(max(i-1,1), j);
            Bt = Brec(i, max(j-1,1));

            % 预测当前像素
            R_pred = r(1)*Rl + r(2)*Rt;
            G_pred = g(3)*Gl + g(4)*Gt ;
            B_pred = b(5)*Bl + b(6)*Bt;

            % 获取真实值
            r_val = Rz(i,j);
            g_val = Gz(i,j);
            b_val = Bz(i,j);

            % 误差量化
            err_r(i,j) = round((r_val - R_pred) / delta);
            err_g(i,j) = round((g_val - G_pred) / delta);
            err_b(i,j) = round((b_val - B_pred) / delta);

            % 用量化误差更新预测图像（递推）
            Rrec(i,j) = R_pred + delta * err_r(i,j);
            Grec(i,j) = G_pred + delta * err_g(i,j);
            Brec(i,j) = B_pred + delta * err_b(i,j);
        end
    end
end