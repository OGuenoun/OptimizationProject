clear all
close all

% Initialisation des données des images
load('Data\DigitTrain_0.mat')
nb_imgs = size(imgs, 3); % nombre d'images
taille = size(imgs, 1) * size(imgs, 2); % nombre de pixels

data = reshape(imgs, [taille, nb_imgs]); % On reforme l'image en vecteur
M(1, 1) = size(imgs, 3); % Nombre d'images pour la classe 0
c = labels.';

for i = 1:9 % 0 ayant déjà été chargé, on commence à 1
    load(['Data/DigitTrain_' num2str(i) '.mat']);
    nb_imgs = size(imgs, 3); % nombre d'images
    taille = size(imgs, 1) * size(imgs, 2); % nombre de pixels

    X_reshape = reshape(imgs, [taille, nb_imgs]);
    M(1, i + 1) = size(imgs, 3);
    data = [data, X_reshape];
    c = [c, labels.'];
end

%---------------------------------------------------------
data = [ones(1, size(data, 2)); data]; % Ajout du biais à la matrice data
N = size(data, 2); % Nombre d'exemples
P = 10; % Nombre de classes
MC = zeros(P, N); % Matrice cible
NbItMax = 350; % Nombre d'itérations

% Définition de la matrice cible
% Va permettre de distinguer les classes
a = 0;
for j = 1:P
    for i = 1:M(j)
        MC(j, a + i) = 1;
    end
    a = a + M(j);
end

% Initialisation des paramètres
rho = 0.1;
w = zeros(size(data, 1), P); % Poids, taille (n x P)
z = zeros(P, N); % z, taille (P x N)
y = zeros(P, N); % y, taille (P x N)
J = zeros(NbItMax, 1); % Critère

% 1. Calcul initial de z et y
z = w' * data; % Calcul de z pour l'itération initiale (taille P x N)
y = 1 ./ (1 + exp(-z)); % Activation sigmoïde pour chaque sortie

% Initialisation de J
J(1) = sum(sum((y - MC).^2)) / (2 * N); % Critère pour les paramètres initiaux

% 2. Descente de gradient
for ind = 2:NbItMax
    for p = 1:P
        % Calcul de z et y pour chaque classe
        z(p, :) = w(:, p)' * data;
        y(p, :) = 1 ./ (1 + exp(-z(p, :)));

        % Calcul du gradient
        deriv1 = (y(p, :) - MC(p, :));
        deriv2 = y(p, :) .* (1 - y(p, :));
        gradJ = (data * (deriv1 .* deriv2)') / N;

        % Mise à jour des paramètres
        w(:, p) = w(:, p) - rho * gradJ;
    end

    % Calcul du critère de coût (J) pour l'itération
    J(ind) = sum(sum((y - MC) .^ 2)) / (2 * N);
end
printf('Partie entrainement OK \n')
%---------------------------------------------------------
% Partie Test

% Initialisation des données de test
load('Data\DigitTest_0.mat')
nb_imgs = size(imgs, 3); % nombre d'images
taille = size(imgs, 1) * size(imgs, 2); % nombre de pixels

datatest = reshape(imgs, [taille, nb_imgs]);
Ntest = size(datatest, 2);
ctest = labels.';

for i = 1:9
    load(['Data/DigitTest_' num2str(i) '.mat']);
    nb_imgs = size(imgs, 3); % nombre d'images
    taille = size(imgs, 1) * size(imgs, 2); % nombre de pixels

    X_reshape = reshape(imgs, [taille, nb_imgs]);
    M(1, i + 1) = size(imgs, 3);
    datatest = [datatest, X_reshape];
    ctest = [ctest, labels.'];
end

datatest = [ones(1, size(datatest, 2)); datatest]; % Ajout du biais

% Initialisation de ztest et ytest
ztest = zeros(P, size(datatest, 2));
ytest = zeros(P, size(datatest, 2));
classe_ytest = ones(1, size(datatest, 2));

% Calcul des éléments de ztest et ytest
for i = 1:P
    ztest(i, :) = w(:, i)' * datatest; % Calcul de ztest pour la classe p
    ytest(i, :) = 1 ./ (1 + exp(-ztest(i, :))); % Calcul de ytest (activation sigmoïde)
end

% Classification : assigner chaque exemple à la classe avec la probabilité maximale
[~, classe_ytest] = max(ytest);
classe_ytest = classe_ytest - 1; % Ajustement pour que les classes soient de 0 à 9

%---------------------------------------------------------
% Génération de la matrice de confusion
matriceConf = confusionmat(ctest, classe_ytest, 'Order', 0:9);

% Affichage de la matrice de confusion
fprintf('Matrice de confusion :\n');
disp(matriceConf);

% Créer une figure pour l'affichage
figure('Name', 'Matrice de Confusion', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);

% Créer et afficher le graphique de confusion
confChart = confusionchart(matriceConf, 0:9);

% Personnaliser l'apparence
confChart.Title = 'Matrice de Confusion pour la Classification des Chiffres Manuscrits';
confChart.XLabel = 'Classe Prédite';
confChart.YLabel = 'Classe Réelle';
confChart.ColumnSummary = 'column-normalized';
confChart.RowSummary = 'row-normalized';

% Précision globale
precision = sum(diag(matriceConf)) / sum(matriceConf(:)) * 100;
xlabel(sprintf('Précision globale : %.2f%%', precision));

