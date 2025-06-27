function RGB_rec = predictionRGB_inv_nocenter(err_r, err_g, err_b, r, g, b, delta, Rmed, Gmed, Bmed)
    [M,N] = size(err_r);

    R_rec = zeros(M,N);
    G_rec = zeros(M,N);
    B_rec = zeros(M,N);

    for i = 1:M
        for j = 1:N
            % 邻域像素（已恢复）
            Rl = R_rec(max(i-1,1), j);
            Rt = R_rec(i, max(j-1,1));
            Gl = G_rec(max(i-1,1), j);
            Gt = G_rec(i, max(j-1,1));
            Bl = B_rec(max(i-1,1), j);
            Bt = B_rec(i, max(j-1,1));

            % 预测
            R_pred = r(1)*Rl + r(2)*Rt;
            G_pred = g(3)*Gl + g(4)*Gt ;
            B_pred = b(5)*Bl + b(6)*Bt;


            % 恢复像素值
            R_rec(i,j) = R_pred + delta * err_r(i,j);
            G_rec(i,j) = G_pred + delta * err_g(i,j);
            B_rec(i,j) = B_pred + delta * err_b(i,j);
        end
    end

%decentralization
R = R_rec + Rmed;
G = G_rec + Gmed;
B = B_rec + Bmed;
RGB_rec = cat(3, min(max(R,0),255), min(max(G,0),255), min(max(B,0),255));
RGB_rec = uint8(RGB_rec);
end
