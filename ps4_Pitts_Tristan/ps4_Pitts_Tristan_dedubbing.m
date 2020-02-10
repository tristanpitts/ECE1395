clear all;

%1a
load 'input/hw4_data1.mat';

%total number of points predicted during each iteration
numpoints = size(y1);

K = 1:2:15;

accuracy = [8, 5];
%{
  accuracy table:

                  accuracy: k=1, k=3, k=5, k=7, k=9, k=11, k=13, k=15
    training set = X1 - X4:
  training set = X1-X3, X5:
training set = X1-X2,X4,X5:
  training set = X1, X3-X5:
       training set: X2-X5:
%}

%Training on X1-X4 to predict X5, testing all k values
for i = 1:8
  KNN = fitcknn([X1; X2; X3; X4], [y1; y2; y3; y4], 'NumNeighbors', K(i));
  [LABEL, POSTERIOR, COST] = predict(KNN, X5);
  accuracy(1, i) = (numpoints - nnz(LABEL - y5)) / numpoints;
end

%Training on X1-X3, X5 to predict X4, testing all k values
for i = 1:8
  KNN = fitcknn([X1; X2; X3; X5], [y1; y2; y3; y5], 'NumNeighbors', K(i));
  [LABEL, POSTERIOR, COST] = predict(KNN, X4);
  accuracy(2, i) = (numpoints - nnz(LABEL - y4)) / numpoints;
end

%Training on X1, X2, X4, X5 to predict X3, testing all k values
for i = 1:8
  KNN = fitcknn([X1; X2; X4; X5], [y1; y2; y4; y5], 'NumNeighbors', K(i));
  [LABEL, POSTERIOR, COST] = predict(KNN, X3);
  accuracy(3, i) = (numpoints - nnz(LABEL - y3)) / numpoints;
end

%Training on X1, X3-X5 to predict X3, testing all k values
for i = 1:8
  KNN = fitcknn([X1; X3; X4; X5], [y1; y3; y4; y5], 'NumNeighbors', K(i));
  [LABEL, POSTERIOR, COST] = predict(KNN, X2);
  accuracy(4, i) = (numpoints - nnz(LABEL - y2)) / numpoints;
end

%Training on X2-X5 to predict X1, testing all k values
for i = 1:8
  KNN = fitcknn([X2; X3; X4; X5], [y2; y3; y4; y5], 'NumNeighbors', K(i));
  [LABEL, POSTERIOR, COST] = predict(KNN, X1);
  accuracy(5, i) = (numpoints - nnz(LABEL - y1)) / numpoints;
end

averageAccuracy = zeros(1, 8);

%calculate average accuracy for each test
for i = 1:8
  averageAccuracy(i) = mean(accuracy(:, i));
end

plot(K, averageAccuracy);
ylabel('Average Accuracy');
xlabel('K Value');
xticks(1:2:15);
saveas(gcf, 'output/ps4-1-a.png');

%2a
load 'input/hw4_data2.mat';

accuracy_weighted = zeros(1, 5);
y_predict = zeros(size(y_test));
numpoints_weighted = size(X_test, 1);
sigma = [0.1, 0.5, 1, 3, 5];

for i=1:5
  y_predict = weightedKNN(X_train, y_train, X_test, sigma(i));
  accuracy_weighted(i) = (numpoints_weighted - nnz(y_test - y_predict)) / numpoints_weighted;
end


T = table(sigma', accuracy_weighted','VariableNames', {'Sigma' 'Accuracy'});
T

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
