function J = computeCost(x, y, theta)

  J=0;

  for i=1:size(x)
    J = J + ((htheta(theta,x(i,:)) - y(i))^2);
  end

  temp = J / (2 * size(x, 1));

  J = temp;
end
