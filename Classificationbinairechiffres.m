% Script pour optimiser le critere par la methode de descente du gradient a
% pas fixe


clear all
close all
clc
load('Data\DigitTest_1.mat')
c=zeros(1,length(imgs))';
X_1=reshape(imgs,[400,length(imgs)]);
load('Data\DigitTest_7.mat')
X_3=reshape(imgs,[400,length(imgs)]);
X=[X_1,X_3];
X=[ones(1,length(X));X];
c=[c;ones(length(imgs),1)]';
N=length(c);
W=zeros(401,1);
Z=zeros(1,N);
Y=zeros(1,N);
% Z=W(:,1).'*X;
% Y=1./(1+exp(-Z));


% Parametres
rho		= 10.^(-1);		% A completer
nbItMax =	250;	% A completer



% 2. Descente de gradient
	% a. Initialisation

		J(1) =(1/(2*N))*(sum(Y-c).^2);

	% b. Iterations
		for ind = 2:nbItMax
            for n=1:401
            gradJ(n,ind-1)=sum((Y-c).*Y.*(1-Y).*X(n,:))/N; % A completer
            end
			% mise a jour des parametres
			W(:,ind)=W(:,ind-1)-rho*gradJ(:,ind-1); % A completer
            %
			for n= 1:N
            	Z(n)=W(:,ind-1).'*X(:,n);
                Y(n)=1/(1+exp(-Z(n)));
             end
			 J(ind)=(1/(2*N))*sum((Y-c).^2);
            %  if J(ind)<J(ind-1)
            %      rho=rho*2;
            %  elseif J(ind)==J(ind-1)
            %      rho=rho*2;
            %  else
            %      rho=rho/2;
            %      W(:,ind)=W(:,ind-1);
            %      J(ind)=J(ind-1);
            %  end
			% Affichage des courbes

		end
	% figure;
	% plot([1:nbItMax],J);
    % xlim([2,N]);
    % figure;
    % subplot(3,1,1);
    % plot([1:nbItMax],W(1,:));
    % subplot(3,1,2);
    % plot([1:nbItMax],W(2,:));
    % subplot(3,1,3);
    % plot([1:nbItMax],W(3,:));
    nbreBon=0;
    classeY=ones(1,N);
    load('Data\DigitTest_1.mat')
    cTest=zeros(1,length(imgs))';
    labels8=labels;
    Xt_1=reshape(imgs,[400,length(imgs)]);
     load('Data\DigitTest_7.mat')
     cTest=[cTest;ones(length(imgs),1)]';
     labels9=labels;
     Xt_3=reshape(imgs,[400,length(imgs)]);
     Xtest=[Xt_1,Xt_3];
     Xtest=[ones(1,length(X));Xtest];
    Nt=length(cTest);
    Ztest=zeros(1,Nt);
    classeYt=ones(1,Nt);

    for n=1:Nt
      Ztest(n)=W(:,nbItMax)'*Xtest(:,n);
    end
    Ytest=1./(1+exp(-Ztest));

    for n=1:Nt
        if Ytest(n)<0.5
            classeYt(n)=0;
            labelsPredict(n)=8;
        end
    end
    for n=1:Nt
        if classeYt(n)==cTest(n)
            nbreBon=nbreBon+1;
        end
    end
    tauxReusssite=nbreBon/N
    Co=confusionmat(cTest,classeYt);
    confusionchart(Co,[1,7]);


