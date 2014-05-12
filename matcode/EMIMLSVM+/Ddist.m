function D = Ddist(X)
%kmeans�е�D���㷨����correlation������
[n,p] = size(X);
D = zeros(n,size(X,1));
nclusts = size(X,1);
normX = sqrt(sum(X.^2, 2));
for i = 1:nclusts
    D(:,i) = max(1 - X * (X(i,:)./normX(i))', 0);
end
