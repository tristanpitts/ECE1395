function [ids, means, ssd] = kmeans_single(X, K, iters)

ids = zeros(size(X, 1), 1);
means = zeros(K, size(X, 2));
ssd = 0;

%random initialization of clusters
for i=1:K
  for j=1:size(X, 2)
    means(i, j) = range(X(:, j))*rand(1);
  end
end

for i=1:iters
  %find min distance cluster for each sample
  for j=1:size(X, 1)

    dist = zeros(K, 1);
    for k=1:K
      dist(k) = pdist2(X(j, :), means(k, :));
    end

    [~, ind] = min(dist);
    ids(j) = ind;

  end

  %recalculate means
  for k=1:K
    temp = X(find(ids == k), :);
    if ~isempty(temp)
      means(k, :) = mean(temp);
    end
  end

end

for i=1:size(X, 2)
  ssd = ssd + pdist2(X(i, :), means(ids(i), :))^2;
end

end
