function [theta, cost] = gradientDescent(x_train, y_train, alpha, iters)

  m = size(x_train, 1);
  n = size(x_train, 2);

  theta = zeros(n, 1);
  cost = zeros(iters, 1);

  for i=1:iters
    cost(i) = computeCost(x_train, y_train, theta);
    %adjust thetas
    for j=1:n
        temp = (theta(j) - alpha*(derivitive_cost(theta, x_train, y_train, j))); %j is which theta derivitive is with respect to
        theta(j) = temp;
    end
  end
end
