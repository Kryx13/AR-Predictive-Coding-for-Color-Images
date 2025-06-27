filename = "Foyer.jpg";
delta = 20;
image = imread(filename);
[r,g,b] = Cal_para2(filename);
[err_r, err_g, err_b, Rmed, Gmed, Bmed] = predictionRGB_nocenter(filename, r, g, b, delta);
reconstructed_image = predictionRGB_inv_nocenter(err_r, err_g, err_b, r, g, b, delta, Rmed, Gmed, Bmed);
err = cat(3, min(max(err_r,0),255), min(max(err_g,0),255), min(max(err_b,0),255));
[mse_rgb, psnr_rgb] = compute_mse_psnr(image, reconstructed_image);

for c = 1:3
        channel = uint8(err(:,:,c));
        h = imhist(channel); p = h / sum(h);
        p = p(p > 0);
        Entropy_vals(c) = -sum(p .* log2(p));

end
fprintf('Entrpy,R=%.2f, G=%.2f, B=%.2f\n', Entropy_vals(1),Entropy_vals(2),Entropy_vals(3));
figure;
imshow(err);
title('err matrix');
figure
imshow(reconstructed_image);
title('prediction');