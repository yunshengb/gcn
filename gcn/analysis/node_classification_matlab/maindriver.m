% this is a sample driver file to show that how to 
% use social dimensions for relationa learning
% Please refer to  [1] and [2] for details.
% 
% [1]: Lei Tang and Huan Liu, Relational Learning via Latent Social Dimensions, KDD, 2009.
% [2]: Lei Tang and Huan Liu. Leveraging Social Media Networks for Classification. Journal of Data Mining and Knowledge Discovery (DMKD), 2011.
% 

rand('seed', 1);


global network;  % decalre network as global variable to save memory space for eigvector computation
                 
%load toy.mat;
load blogcatalog.mat



% compute the social dimensions via modualrity maximiztion
k = 500;        % number of social dimensions to extract
[V, d] = modularity(k); 

% uncomment the following lines if you want to use spectral clustering
% k = 500;
% [V, d] = spectralclustering(network, k);

% uncomment the following lines if you want to use EdgeCluster
% k=5000; % set overlapping clusters to 5000
% [V] = edgeclustering(network, k);


% randomly generate index_tr and index_te
n = size(network, 1);
index = randperm(n);
index_tr = index(1:ceil(0.9*n));  % 90% labeled nodes for training
index_te = index(1+ceil(0.9*n):end);  % 10% unlabeled nodes for test

% bulild the classifier and make predictions
C = 500; % the C parameter in SVM Classifier
labels = group(index_tr, :); % the labels of nodes for training
[predscore] = SocioDim(V, labels, index_tr, index_te, C);

[perf, pred] = evaluate(predscore, group(index_te, :));
disp('Driver program ends!');
perf