function [V, d] = spectralclustering(G, k)
% Given a symmetric graph (or similarity matrix), perform spectral clustering on normalized graph Laplacian.
% 
% function [V, d] = spectralclustering(G, k)
% Input: 
%  - G: a symmetric graph (or similarity matrix of size n X n);
%  - k: number of clusters;
%
% Output: 
%  - V: the smallest eigenvectors of the graph Laplacian
%  - d: corresponding eigenvalues
% 
% Code provided by Lei Tang in July, 2009. 
%

n = size(G, 1);

% set the diagonal to zero
G = G- sparse(1:n, 1:n, diag(G));


d = full(sum(G));
dhalf = 1./sqrt(d);

% find out those nodes without connections
dhalf(d==0) = 0;

L = sparse(1:n, 1:n, 1) - sparse(1:n, 1:n, dhalf) * G * sparse(1:n, 1:n, dhalf);

[V, D] = eigs(L, k, 'SA');

d = diag(D);

