function [ids, means, ssd] = kmeans_multiple(X, K, iters, R)

for i=1:R

  [tempIds, tempMeans, tempSsd] = kmeans_single(X, K, iters);

  if(i == 1)
    ids = tempIds;
    means = tempMeans;
    ssd = tempSsd;
  else if(tempSsd < ssd)
    ids = tempIds;
    means = tempMeans;
    ssd = tempSsd;
  end

end

end
