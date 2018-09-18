% -- Anirban Bhattacharya -- %

% Gibbs sampler for covariance estimation
% using mgps prior on factor loadings

% lambda_{jh} \sim N(0, psi_{jh}^{-1} tau_h^{-1})


clear;clc;close all; tic;

load datagenp_30ktr_5rep_1;

% --- define global constants --- %
nrun=20000; burn=5000; thin=1; sp =(nrun - burn)/thin; % number of posterior samples

kinit = repmat(floor(log(p)*3),rep,1);                 % number of factors to start with
b0 = 1; b1 = 0.0005;
epsilon = 1e-3;                                        % threshold limit
prop = 1.00;                                           % proportion of redundant elements within columns


%---- define output files across replicates-----%

mserep = zeros(rep,3);                          % mse,absolute bias(avg and maximum) in estimating cov matrix
mse1rep = zeros(rep,3);                         % same as above in original scale in estimating cov matrix
nofrep = zeros(rep,sp);                         % evolution of factors across replicates
adrep = zeros(rep,1);                           % number of adaptations across replicates

for g = 1:rep

    disp(['start replicate','',num2str(g)]);
    disp('--------------------');

    % ------read data--------%
    Y = dat((g-1)*n + 1:g*n,:);                 % n x p data matrix
    M=mean(Y);VY=var(Y);
    Y = bsxfun(@minus,Y,M);                     % center the training data
    Y = bsxfun(@times,Y,1./sqrt(VY));           % scale the training data

    Ot1 = Ot.*(1./sqrt(VY'*VY));                % true dispersion matrix of the transformed data
    num = 0;
    k=kinit(g);                                 % no. of factors to start with

    % ------end read data--------%

    % --- Define hyperparameter values --- %
    
    as = 1;bs = 0.3;                           % gamma hyperparameters for residual precision
    df = 3;                                    % gamma hyperparameters for t_{ij}
    ad1 = 2.1;bd1 = 1;                         % gamma hyperparameters for delta_1
    ad2 = 3.1;bd2 = 1;                         % gamma hyperparameters delta_h, h >= 2
    adf = 1; bdf = 1;                          % gamma hyperparameters for ad1 and ad2 or df

    % --- Initial values --- %
    ps=gamrnd(as,1/bs,p,1); Sigma=diag(1./ps);                  % Sigma = diagonal residual covariance
    Lambda = zeros(p,k);  ta =  normrnd(0,1,[n,k]);             % factor loadings & latent factors
    meta = zeros(n,k); veta = eye(k);                           % latent factor distribution = standrad normal
                                
    psijh = gamrnd(df/2,2/df,[p,k]);                            % local shrinkage coefficients
    delta = ...
        [gamrnd(ad1,bd1);gamrnd(ad2,bd2,[k-1,1])];              % gobal shrinkage coefficients multilpliers
    tauh = cumprod(delta);                                      % global shrinkage coefficients
    Plam = bsxfun(@times,psijh,tauh');                          % precision of loadings rows

    % --- Define output files specific to replicate --- %
    nofout = zeros(nrun+1,1);                  % number of factors across iteartions
    nofout(1) = k;
    nof1out = zeros(sp,1);
    mseout = zeros(sp,3);                      % within a replicate, stores mse across mcmc iterations
    mse1out = zeros(sp,3);                     % mse in original scale
    Omegaout = zeros(p^2,1);
    Omega1out = zeros(p^2,1);




    %------start gibbs sampling-----%

    for i = 1:nrun

        % -- Update eta -- %
        Lmsg = bsxfun(@times,Lambda,ps);
        Veta1 = eye(k) + Lmsg'*Lambda;
        T = cholcov(Veta1); [Q,R] = qr(T);
        S = inv(R); Veta = S*S';                   % Veta = inv(Veta1)
        Meta = Y*Lmsg*Veta;                        % n x k 
        eta = Meta + normrnd(0,1,[n,k])*S';        % update eta in a block

        % -- update Lambda (rue & held) -- %
        eta2 = eta'*eta;
        for j = 1:p
            Qlam = diag(Plam(j,:)) + ps(j)*eta2; blam = ps(j)*(eta'*Y(:,j));
            Llam = chol(Qlam,'lower'); zlam = normrnd(0,1,k,1);
            vlam = Llam\blam; mlam = Llam'\vlam; ylam = Llam'\zlam;
            Lambda(j,:) = (ylam + mlam)';
        end

        %------Update psi_{jh}'s------%
        psijh = gamrnd(df/2 + 0.5,1./(df/2 + bsxfun(@times,Lambda.^2,tauh')));

        %------Update delta & tauh------%
        mat = bsxfun(@times,psijh,Lambda.^2);
        ad = ad1 + 0.5*p*k; bd = bd1 + 0.5*(1/delta(1))*sum(tauh.*sum(mat)');
        delta(1) = gamrnd(ad,1/bd);
        tauh = cumprod(delta);

        for h = 2:k
            ad = ad2 + 0.5*p*(k-h+1); bd = bd2 + 0.5*(1/delta(h))*sum(tauh(h:end).*sum(mat(:,h:end))');
            delta(h) = gamrnd(ad,1/bd); tauh = cumprod(delta);
        end

        % -- Update Sigma -- %
        Ytil = Y - eta*Lambda'; 
        ps=gamrnd(as + 0.5*n,1./(bs+0.5*sum(Ytil.^2)))';
        Sigma=diag(1./ps);

        %---update precision parameters----%
        Plam = bsxfun(@times,psijh,tauh');

        % ----- make adaptations ----%
        prob = 1/exp(b0 + b1*i);                % probability of adapting
        uu = rand;
        lind = sum(abs(Lambda) < epsilon)/p;    % proportion of elements in each column less than eps in magnitude
        vec = lind >=prop;num = sum(vec);       % number of redundant columns

        if uu < prob
            if  i > 20 && num == 0 && all(lind < 0.995)
                k = k + 1;
                Lambda(:,k) = zeros(p,1);
                eta(:,k) = normrnd(0,1,[n,1]);
                psijh(:,k) = gamrnd(df/2,2/df,[p,1]);
                delta(k) = gamrnd(ad2,1/bd2);
                tauh = cumprod(delta);
                Plam = bsxfun(@times,psijh,tauh');
            elseif num > 0
                nonred = setdiff(1:k,find(vec)); % non-redundant loadings columns
                k = max(k - num,1);
                Lambda = Lambda(:,nonred);
                psijh = psijh(:,nonred);
                eta = eta(:,nonred);
                delta = delta(nonred);
                tauh = cumprod(delta);
                Plam = bsxfun(@times,psijh,tauh');
            end
        end
        nofout(i+1) = k;

        % -- save sampled values (after thinning) -- %
        if mod(i,thin)==0 && i > burn
            Omega = Lambda*Lambda' + Sigma;
            Omega1 = Omega .* sqrt(VY'*VY);
            Omegaout = Omegaout + Omega(:)/sp;
            Omega1out = Omega1out + Omega1(:)/sp;
            nof1out((i-burn)/thin) = (nofout((i-burn)/thin) - num)*(num > 0);
        end

        if mod(i,1000) == 0
            disp(i);
        end
    end


    %---- summary measures specifi to replicate ---%
    %1. covariance matrix estimation
    errcov = Omegaout - Ot1(:); err1cov = Omega1out - Ot(:);
    mserep(g,:) = [mean(errcov.^2) mean(abs(errcov)) max(abs(errcov))];
    mse1rep(g,:) = [mean(err1cov.^2) mean(abs(err1cov)) max(abs(err1cov))];

    %w. evolution of factors
    nofrep(g,:) = nof1out';

    disp(['end replicate','',num2str(g)]);
    disp('--------------------');



end

toc;

