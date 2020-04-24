function g_prime = sigmoidGradient(z)

g_prime = zeros(size(z, 1), 1);

for i=1:size(z, 1)

  g_prime(i) = sigmoid(z(i)) * (1 - sigmoid(z(i)));

end
