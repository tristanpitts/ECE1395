function p = predict(theta1, theta2, x)

p = zeros(size(x, 1), 1);
a = ones(size(x, 1), 1);
x = [a x];

z1 = x*theta1';

a2 = sigmoid(z1);
a2 = [a a2];

z2 = a2*theta2';
a3 = sigmoid(z2);

for i=1:size(x, 1)

  [value, argmax] = max(a3(i, :));
  p(i) = argmax;

end
