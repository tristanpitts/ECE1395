clear all;

load 'input/HW7_data1.mat';
load 'input/HW7_weights_1.mat';

indecies = zeros(1, 20);
for i=1:16
  temp = randi([1, 5000], 1);
  while(any(indecies==temp))
    randi([1, 5000], 1);
  end
  indecies(i) = temp;
end

for i=1:16
  subplot(4, 4, i)
  imagesc(reshape(X(indecies(i), :), [20, 20]));
  axis off;
end

saveas(gcf, 'output/ps7-1-b.png');

p = predict(Theta1, Theta2, X);

accuracy = (5000 - nnz(y - p))/5000;
accuracy

sigmoidGradient([-10, 0, 10]')

nn_params = [Theta1(:); Theta2(:)];
fprintf("Lambda = 0\n")
nnCostFunction(nn_params, 400, 25, 10, X, y, 0)
fprintf("Lambda = 1\n");
nnCostFunction(nn_params, 400, 25, 10, X, y, 1)

checkNNGradients

fprintf("Lambda=3\n");
checkNNGradients(3);

lambdaValues = [0, 1, 2, 4];
maxIter = [50, 100, 200, 400];
accuracy = zeros(4, 4);
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

for i=1:4
  for j=1:4

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);

    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    options = optimset('MaxIter', maxIter(i));
    lambda = lambdaValues(j);

    costFunction = @(p) nnCostFunction(nn_params, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, ...
                                       X, y, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    p = predict(Theta1, Theta2, X);
    accuracy(j, i) = (5000 - nnz(y - p))/5000;
  end
end

rows = {'lambda=0', 'lambda=1', 'lambda=2', 'lambda=4'};
columns = {'MaxIter_50','MaxIter_100','MaxIter_200','MaxIter_400'};

T = array2table(accuracy, 'RowNames', rows, 'VariableNames', columns);

T
