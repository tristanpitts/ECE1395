function y_predict = weightedKNN(X_train, y_train, X_test, sigma)

y_predict = zeros(size(X_test, 1), 1);

for i=1:size(X_test, 1)
  weight = zeros(size(unique(y_train),1) , 1);
  for j=1:size(X_train, 1)
    weight(y_train(j)) = weight(y_train(j)) + exp(-1*pdist2(X_train(j), X_test(i))/(sigma^2));
  end
  y_predict(i) = find(weight==max(weight));
end

end
