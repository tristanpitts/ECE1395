function dc = derivitive_cost(theta, x, y, j)

dc = 0;

  for i=1:size(x, 1)
    dc = dc + (htheta(theta, x(i, :)) - y(i)) * x(i,j);
  end
  temp = dc / size(x,1);
  dc = temp;
end
