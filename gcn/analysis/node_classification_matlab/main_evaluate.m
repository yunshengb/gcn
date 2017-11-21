features = load('blog_200d_embedding.mat');
features = single(features.embedding);
% features = normr(features);
%group = load('blogcatalog.mat');
%group = group.arr;

V = features;
load blogcatalog;
n = size(network, 1);
macrof1 = size(1,5);
microf1 = size(1,5);

for t = 1: 5
    index = randperm(n);
    index_tr = index(1:ceil(0.9*n));  % 90% labeled nodes for training
    index_te = index(1+ceil(0.9*n):end);  % 10% unlabeled nodes for test

    % bulild the classifier and make predictions
    C = 500; % the C parameter in SVM Classifier
    labels = group(index_tr, :); % the labels of nodes for training
    [predscore] = SocioDim(V, labels, index_tr, index_te, C);
    % save predscore;
    tlabels = group(index_te, :);
    % save tlabels;
    [perf, pred] = evaluate(predscore, tlabels);
    disp(t);
    macrof1(t) = perf.macro_F1;
    microf1(t) = perf.micro_F1;
end

micro_f1 = mean(microf1)
macro_f1 = mean(macrof1)
