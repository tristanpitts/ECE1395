function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by following the procedure highlighted in the assignment

%from predict.m
a = ones(size(X, 1), 1);
x = [a X];

z1 = x*Theta1';

a2 = sigmoid(z1);
a2 = [a a2];

z2 = a2*Theta2';
a3 = sigmoid(z2);

translate_output = zeros(m, num_labels);

for i=1:m
  translate_output(i, y(i)) = 1;
end

J = 0;

for i=1:m
  for k=1:num_labels
    J = J - translate_output(i, k)*log(a3(i,k)) - (1-translate_output(i, k))*log(1-a3(i,k));
  end
end

reg = 0;

for j=1:size(Theta1, 1)
  for k=2:size(Theta1, 2)-1
    reg = reg + Theta1(j, k)^2;
  end
end

for j=1:size(Theta2, 1)
  for k=2:size(Theta2, 2)-1
    reg = reg + Theta2(j, k)^2;
  end
end
reg = reg*lambda / (2*m);
J = reg + J/m;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

for i=1:m
  delta3 = a3(i, :) - translate_output(i, :);
  Theta2_grad = Theta2_grad + delta3'*a2(i, :);

  temp = [ones(size(z2, 1), 1) z2];
  delta2 = (delta3*Theta2).*sigmoidGradient(temp(i, :));
  delta2 = delta2(:, 2:end);
  Theta1_grad = Theta1_grad + delta2'*x(i, :);
end

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
