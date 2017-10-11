function [V, d] = modularity(k)
% [V, d] = modularity(k)
% calculate the top eigenvectors of modularity matrix B
% The network matrix NETWORK is decalred to be global to save space
% network is assumed to be symmetric
% 
% INPUT:
%  - k: number of top eigenvectors to compute
%  - network: a sparse matrix representing the network, a GLOBAL variable
% OUTPUT:
%  - V: the top eigenvectors
%  - d: the corresponding eigenvalues
%
% Updated by Lei Tang on Sep. 23rd, 2009.

global network;
d = sum(network,2); % d is a col vector
twom = sum(d);      % the total number of degrees
opts.issym = 1;
opts.isreal = 1;

n = size(network, 1);
[V, D] = eigs(@(x)matrix_vector_multiplication(x), n, k, 'LA', opts);
d = diag(D);

function [res] = matrix_vector_multiplication(x)
% A innate function implementing the matrix / vector multiplication
  res = network*x - (d'*x)/twom * d;
end
end
