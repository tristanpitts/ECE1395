function h = htheta(x, theta)

  h = sigmoid(sum(theta'.*x));

end
