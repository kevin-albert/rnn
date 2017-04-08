n = 10;
density = 0.1;
alpha = 0.9;

% Length of sparse matrix initializers
len = max(1,floor(n*n*density));

% Diagonal matrix of random eigenvalues
D = diag((rand(1,n)*2-1)*alpha);

% Random eigenvectors
V = zeros(n,n);
while det(V) == 0
    
    i = randperm(n);
    j = randperm(n);
    
    V = sparse(i(1:len), j(1:len), rand(len,1)*2-1,n,n);
end

X = V*D*inv(V);