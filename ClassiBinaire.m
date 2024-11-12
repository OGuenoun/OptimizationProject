% Script pour optimiser les poids  pour les donées simulées avec  la methode de descente du gradient a pas fixe


 clear all
 close all
 clc
load('DataSimulation\DataTrain_2Classes_Perceptron_2.mat')
N=length(c);
Z=zeros(1,N);
Y=zeros(1,N);
X=[ones(1,N);data];




% Parametres
rho		= 2;		% A completer
nbItMax =	1000;	% A completer
W=zeros(3,nbItMax);
W(:,1)=[0.25;0.25;0.25];


% 2. Descente de gradient
	% a. Initialisation
     for n= 1:N
            	Z(n)=W(:,1).'*X(:,n);
                Y(n)=1/(1+exp(-Z(n)));
             end
		J(1) =(1/(2*N))*(sum(Y-c).^2);

	% b. Iterations
		for ind = 2:nbItMax

            gradJ(1,ind-1)=sum((Y-c).*Y.*(1-Y).*X(1,:))/N; % A completer
			gradJ(2,ind-1)=sum((Y-c).*Y.*(1-Y).*X(2,:))/N;
            gradJ(3,ind-1)=sum((Y-c).*Y.*(1-Y).*X(3,:))/N;
			% mise a jour des parametres
			W(:,ind)=W(:,ind-1)-rho*gradJ(:,ind-1); % A completer

			for n= 1:N
            	Z(n)=W(:,ind-1).'*X(:,n);
                Y(n)=1/(1+exp(-Z(n)));
            end
			J(ind)=(1/(2*N))*sum((Y-c).^2);
%             if J(ind)<J(ind-1)
%                 rho=rho*2;
%             elseif J(ind)==J(ind-1)
%                 rho=rho*2;
%             else
%                 rho=rho/2;
%                 W(:,ind)=W(:,ind-1);
%                 J(ind)=J(ind-1);
%            end

  end
  % Affichage des courbes

    figure(2);
    subplot(4,1,1);
    plot([1:nbItMax],W(1,:));
    title('w0 en fonction des itérations','fontsize',16);
    subplot(4,1,2);
    plot([1:nbItMax],W(2,:));
    title('w1 en fonction des itérations','fontsize',16);
    subplot(4,1,3);
    plot([1:nbItMax],W(3,:));
    title('w2 en fonction des itérations','fontsize',16);
    subplot(4,1,4);
    plot([1:nbItMax],J,'r');
    title('la valeur du critère en fonction des itérations','fontsize',16);
    nbreBon=0;
    classeYt=ones(1,N);
    load('DataSimulation\DataTest_2Classes_Perceptron_2.mat')
    % Applique les données de tests au poids optimisés

    Xtest=[ones(1,N);dataTest];
    Ztest=zeros(1,N);
    for n= 1:N
      Ztest(n)=W(:,nbItMax)'*(Xtest(:,n));
      end
    Ytest=1./(1+exp(-Ztest));

    for n=1:N
        if Ytest(n)<0.5
            classeYt(n)=0;
        end
    end
    % Calcul du taux de bonne classification

    for n=1:N
        if classeYt(n)==cTest(n)
            nbreBon=nbreBon+1;
        end
    end
    tauxReusssite=nbreBon/N
    %Tracé de la valeur du sortie du perceptron

    figure(1);
    image_cout(1,W(:,nbItMax));
     hold on
    plot(dataTest(1,:),dataTest(2,:),'+g')
    legend('data de test')
    hold on
    plot(data(1,:),data(2,:),'+r',"displayname","data d entrainement")
