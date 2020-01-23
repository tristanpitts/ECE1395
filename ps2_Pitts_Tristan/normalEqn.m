function [ theta ] = normalEqn(x_train, y_train)

  theta = pinv(x_train'*x_train)*x_train'*y_train;

end
