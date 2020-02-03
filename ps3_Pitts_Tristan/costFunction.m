function [J, grad] = costFunction(theta, x_train, y_train)

  J = 0;
  grad = zeros(size(theta));

  for i=1:size(x_train, 1)
    J = J - (y_train(i)*log(htheta(x_train(i, :), theta))) - ((1-y_train(i))*log(1 - htheta(x_train(i, :), theta)));
  end

  for j = 1:size(theta, 1)
    for i = 1:size(x_train, 1)
      grad(j) = grad(j) + gradientCost(x_train(i, :), y_train(i), theta, j);
    end
  end

  J = J/size(y_train, 1);
  grad = grad./size(y_train, 1);

end
