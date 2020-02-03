function grad = gradientCost(x, y, theta, j)

  grad = (htheta(x, theta) - y)*x(j);

end
