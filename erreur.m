function erreur = calculerMatriceErreur(imageReelle, imagePredite)
% CALCULERMATRICEERREUR Calcule la matrice d'erreur entre deux images RGB 
% représentées sous forme de cellules (chaque cellule contient [R G B]) /  Calculates the error matrix between two RGB images 
% represented as cells (each cell contains [R G B])
%
%   erreur = calculerMatriceErreur(imageReelle, imagePredite)
%
%   Output :
%       erreur : matrice HxW de cellules contenant [dR dG dB] / error: HxW matrix of cells containing [dR dG dB].

    [H, W] = size(imageReelle);

    % Vérification des dimensions
    if ~isequal(size(imageReelle), size(imagePredite))
        error('Les dimensions des deux images doivent être identiques.');
    end

    erreur = cell(H, W);

    for i = 1:H
        for j = 1:W
            pixelReel = double(imageReelle{i, j});
            pixelPrevu = double(imagePredite{i, j});
            erreur{i, j} = pixelReel - pixelPrevu; % Donne [dR dG dB]
        end
    end
end

function H = calc_entropie(image)
% CALC_ENTROPIE Calcule l'entropie d'une image (grayscale ou RGB) / Calculates the entropy of an image 
%
%   H = calc_entropie(image)
%
%   Input :
%       image : matrice niveaux de gris RGB / image: RGB grayscale matrix
%
%   Output :
%       H : entropie (en bits) / entropy

    % Si RGB, convertir en niveaux de gris
    if ndims(image) == 3
        image = rgb2gray(image);
    end

    % Convertir en vecteur de pixels
    image = double(image(:));

    % Obtenir histogramme normalisé
    nbNiveaux = 256;
    h = histcounts(image, 0:nbNiveaux, 'Normalization', 'probability');

    % Supprimer les zéros pour éviter log2(0)
    h(h == 0) = [];

    % Calcul de l'entropie
    H = -sum(h .* log2(h));
end

