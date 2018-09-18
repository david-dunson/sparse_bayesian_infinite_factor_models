% Anirban Bhattacharya

% generate covariance matrix from a factor structure 
% with sparse loadings


clear; clc; 

% n = # samples, p = # vars.
% k = # factors, rep = # simulation replictes
rep = 1;n = 200; N = n*rep;
p = 30; k = 5;   

% Lambda = loadings, numeff = # non-zero entries in each column of Lambda
% numeff varies between (k+1) to 2*k

Lambda = zeros(p,k);
numeff = k + randperm(k); 

% generate loadings
for h = 1:k
    temp = randsample(p,numeff(h));
    Lambda(temp,h) = normrnd(0,1,[numeff(h),1]);
end

% generative model: N(0, Lambda Lambda' + sigma^2 I)
mu = zeros(1,p);
Ot = Lambda*Lambda' + 0.01* eye(p);

% dat: N x p data matrix
% rows of dat: random vectors drawn from covariance Ot
dat =  mvnrnd(mu,Ot,N); 

ktr = k; rktr = rank(Lambda); Lamtr = Lambda;

save(strcat('datagenp_',num2str(p),'ktr_',num2str(k),'rep_',num2str(rep)),...
    'dat','Ot','rep','n','p','ktr','rktr','Lamtr');

 