function D = Ddist(X)
%kmeans中的D的算法，按correlation距离算
[n,p] = size(X);
D = zeros(n,size(X,1));
nclusts = size(X,1);
normX = sqrt(sum(X.^2, 2));
for i = 1:nclusts
    D(:,i) = max(1 - X * (X(i,:)./normX(i))', 0);
end
