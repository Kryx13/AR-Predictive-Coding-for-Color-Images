function erreur = calculerMatriceErreur(imageReelle, imagePredite)


    if ~isequal(size(imageReelle), size(imagePredite))
        error('Les images doivent avoir la même taille.');
    end

    erreur = double(imageReelle) - double(imagePredite);
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

