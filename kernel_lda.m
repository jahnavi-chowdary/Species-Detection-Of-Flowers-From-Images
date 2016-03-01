function [eigvector, eigvalue, elapse] = kernel_lda(options,gnd,data)           

if ~exist('data','var')
    global data;
end

if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'Regu') | ~options.Regu
    bPCA = 1;
else
    bPCA = 0;
    if ~isfield(options,'ReguAlpha')
        options.ReguAlpha = 0.01;
    end
end


if isfield(options,'Kernel') & options.Kernel
    K = data;
    clear data;
    K = max(K,K');
    elapse.timeK = 0;
else
    [K, elapse.timeK] = constructKernel(data,[],options);
end



tmp_T = cputime;

% ====== Initialization
nSmp = size(K,1);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

K_orig = K;

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

%======================================
% SVD
%======================================

if bPCA  
    
    [U,D] = eig(K);
    D = diag(D);
    
    maxEigValue = max(abs(D));
    eigIdx = find(abs(D)/maxEigValue < 1e-6);
    if length(eigIdx) < 1
        [dump,eigIdx] = min(D);
    end
    D (eigIdx) = [];
    U (:,eigIdx) = [];
    
    elapse.timePCA = cputime - tmp_T;
    tmp_T = cputime;

    Hb = zeros(nClass,size(U,2));
    for i = 1:nClass,
        index = find(gnd==classLabel(i));
        classMean = mean(U(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end

    [dumpVec,eigvalue,eigvector] = svd(Hb,'econ');
    eigvalue = diag(eigvalue);

    if length(eigvalue) > Dim
       eigvalue = eigvalue(1:Dim);
       eigvector = eigvector(:,1:Dim);
    end
    
    eigvector =  (U.*repmat((D.^-1)',nSmp,1))*eigvector;
    
else
    
    Hb = zeros(nClass,nSmp);
    for i = 1:nClass,
        index = find(gnd==classLabel(i));
        classMean = mean(K(index,:),1);
        Hb (i,:) = sqrt(length(index))*classMean;
    end
    B = Hb'*Hb;
    T = K*K;
    
    elapse.timePCA = cputime - tmp_T;
    tmp_T = cputime;

    for i=1:size(T,1)
        T(i,i) = T(i,i) + options.ReguAlpha;
    end

    B = double(B);
    T = double(T);
    
    B = max(B,B');
    T = max(T,T');
    
    option = struct('disp',0);
    [eigvector, eigvalue] = eigs(B,T,Dim,'la',option);
    eigvalue = diag(eigvalue);
end

   
tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1); 


elapse.timeMethod = cputime - tmp_T; 
elapse.timeAll = elapse.timeK + elapse.timePCA + elapse.timeMethod;
    
