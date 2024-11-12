clear all
close all
load('DataSimulation/DataTrain_2Classes_Perceptron_2.mat');

N=length(data); % On enleve le biais
data = [ones(1, N); data]; % Ajout du biais à la matrice data
P = 2; % Nombre de classes
MC = zeros(P, N); % Matrice cible
nbItMax = 1500; % Nombre d'itérations
L= 15; % nombre de neurones de la couche cachée

c=c+1;
% Remplissage de la matrice cible avec un encodage One-Hot
for n = 1:N
    MC(c(n), n) = 1; % Met 1 dans la ligne correspondant à la classe
    % avec c(n) la Classe de l'exemple n (supposée être entre 1 et P)
end

% Initialisation des paramètres
#w1 = randn(3,L)*sqrt(2/3); % Initialisation He pour la première couche
#w2 = randn(L,P)*sqrt(2/L); % Initialisation He pour la deuxième couche
rho = 12;
w1 = zeros(3,L);
w1(:,1)=[7.7044,1.9583,0.16956];
z1 = zeros(L,N);
y1 = zeros(L,N);

w2 = zeros(L,P); %taille 2*15
w2(:,1)=[-4.1545,-6.165,0.081463,-7.0899,2.0582,6.2843,7.0354,4.0291,4.5846,-7.7424,-2.9155,6.0034,5.7462,0.70189,-6.1575];
z2 = zeros(P,N); %taille 15*2000
y2 = zeros(P,N);%taille 15*2000

% Propagation avant pour la couche cachée
z1 = w1.'* data; % Calcul de z1 (L x N)
y1 = 1 ./ (1 + exp(-z1));

% Propagation avant pour la couche de sortie
z2 = w2.'* y1; % Calcul de z2 (P x N)
y2 = 1 ./ (1 + exp(-z2)); % Activation sigmoïde pour la couche de sortie (P x N)

% Initialisation de J
J2(1) = sum(sum((y2 - MC).^2)) / (2 * N); % Critère pour les paramètres initiaux


% 2. Descente de gradient
for ind = 2:nbItMax;
			% Calcul de z et y
        z1 = w1.' * data;
        y1 = 1 ./ (1 + exp(-z1));

        z2 = w2.' * y1;
        y2 = 1 ./ (1 + exp(-z2));


        % Calcul de l'erreur de la couche de sortie
        delta2 = (y2 - MC) .* y2 .* (1 - y2); % Taille (P, N)

        % Calcul de l'erreur de la couche cachée
        delta1 = (w2 * delta2) .* y1 .* (1 - y1); % Taille (L, N)

        % Calcul des gradients
        gradJ2 = (y1 * delta2') / N; % Taille (L, P)
        gradJ1 = (data * delta1') / N; % Taille (I_0 + 1, L)

        % Mise à jour des poids
        w1 = w1 - rho * gradJ1;
        w2 = w2 - rho * gradJ2;


    % Calcul du critère de coût (J) pour l'itération
    J2(ind) = sum(sum((y2 - MC) .^ 2)) / (2 * N);
end



printf('Partie entrainement OK \n')

% Ajoutez après la boucle d'entraînement
figure;
plot(J2);
title('Évolution du coût pendant l''entraînement');
xlabel('Itération');
ylabel('Coût');

% Partie Test
load('DataSimulation\DataTest_2Classes_Perceptron_2.mat');

% Initialisation de ztest et ytest
ztest1 = zeros(L,N);
ytest1 = zeros(L,N);

ztest2 = zeros(P,N);
ytest2 = zeros(P,N);

datatest = [ones(1, N); dataTest]; % Ajout du biais

% Calcul des éléments de ztest et ytest
ztest1= w1.' * datatest; % Calcul de ztest pour la classe p
ytest1= 1 ./ (1 + exp(-ztest1)); % Calcul de ytest (activation sigmoïde)

ztest2= w2.' * ytest1; % Calcul de ztest pour la classe p
ytest2= 1 ./ (1 + exp(-ztest2)); % Calcul de ytest (activation sigmoïde)

##% Ajoutez ce code après le calcul de ytest2
##figure;
##subplot(2,1,1);
##plot(ytest2(1,:), 'b.'); hold on;
##plot(ytest2(2,:), 'r.');
##title('Probabilités de sortie pour chaque classe');
##legend('Classe 0', 'Classe 1');
##ylabel('Probabilité');
##xlabel('Échantillon');
##
##subplot(2,1,2);
##histogram(ytest2(1,:), 20, 'Normalization', 'probability');
##hold on;
##histogram(ytest2(2,:), 20, 'Normalization', 'probability');
##title('Distribution des probabilités');
##legend('Classe 0', 'Classe 1');


% Classification : assigner chaque exemple à la classe avec la probabilité maximale
[~, classe_ytest] = max(ytest2);
classe_ytest = classe_ytest-1 ;

%---------------------------------------------------------
% Génération de la matrice de confusion
matriceConf = confusionmat(cTest, classe_ytest, 'Order', [0 1]);

% Affichage de la matrice de confusion
fprintf('Matrice de confusion :\n');
disp(matriceConf);

% Affichage de la matrice de confusion sous forme graphique
figure('Name', 'Matrice de Confusion', 'NumberTitle', 'off');
imagesc(matriceConf);
colormap(jet);
colorbar;

% Annotation des cellules
textStrings = num2str(matriceConf(:), '%d');  % Convertir les valeurs en texte
textStrings = strtrim(cellstr(textStrings));  % Supprimer les espaces inutiles
[x, y] = meshgrid(1:2);  % Coordonnées pour chaque cellule
text(x(:), y(:), textStrings, 'HorizontalAlignment', 'center', 'Color', 'white');

% Paramètres de l'affichage
title('Matrice de Confusion');
xlabel('Classe Prédite');
ylabel('Classe Réelle');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Classe 0', 'Classe 1'}, 'YTick', 1:2, 'YTickLabel', {'Classe 0', 'Classe 1'});
axis square;

% Précision globale
precision = sum(diag(matriceConf)) / sum(matriceConf(:)) * 100;
fprintf('Précision globale : %.2f%%\n', precision);
