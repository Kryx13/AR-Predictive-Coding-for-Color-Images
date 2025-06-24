function erreur = calculerMatriceErreur(imageReelle, imagePredite)
% CALCULERMATRICEERREUR Calcule la matrice d'erreur entre deux images RGB 
% représentées sous forme de cellules (chaque cellule contient [R G B])
%
%   erreur = calculerMatriceErreur(imageReelle, imagePredite)
%
%   Output :
%       erreur : matrice HxW de cellules contenant [dR dG dB]

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
