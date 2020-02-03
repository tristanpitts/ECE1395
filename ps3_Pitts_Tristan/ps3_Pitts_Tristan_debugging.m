function f = ps3()

clear all;

%1a
A = csvread('input/hw3_data1.txt');

X = [ones(size(A, 1), 1)];
X(:, 2) = A(:, 1);
X(:, 3) = A(:, 2);
Y = A(:, 3);


fprintf('The size of the feature matrix x is [%d, %d] and the size of the label vector y is [%d, %d]\n', size(X), size(Y))

%1b
gscatter(X(:,2), X(:,3), Y, 'rk', 'o+', 6)
xlim([30 100])
ylim([30 100])
xlabel('Exam 1')
ylabel('Exam 2')
legend('Not Admitted', 'Admitted', 'Location', 'southwest')
saveas(gcf, 'output/ps3-1-b.png')

%1c
g = -10:1:10;
gz = [];

for i=1:size(g,2)
  gz(i) = sigmoid(g(i));
end

plot(g, gz)
xlabel('g')
ylabel('gz')
saveas(gcf, 'output/ps3-1-c.png')

%1d
theta = [0; 0; 0];
cost = costFunction(theta, X, Y)
fprintf("The cost when theta = [0, 0, 0]' is %f\n", cost);

%1e
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t) (costFunction(t, X, Y)), theta, options);

fprintf("The optimal parameters given by the fminunc function are theta(1) = %f, theta(2) = %f and theta(3) = %f\n", theta)
fprintf("The cost value at convergence is %f\n", cost)

%1f
boundary1 = [-1*theta(1)/theta(2), 0];
boundary2 = [0, -1*theta(1)/theta(3)];
gscatter(X(:,2), X(:,3), Y, 'rk', 'o+', 6)
xlim([30 100])
ylim([30 100])
xlabel('Exam 1')
ylabel('Exam 2')
hold on
legend('Not Admitted', 'Admitted', 'Decision Boundary')
plot(boundary1, boundary2)
hold off
saveas(gcf, 'output/ps3-1-f.png')

%1g
fprintf("The probablility of the student being admitted is %f\n", sigmoid(theta(1) + theta(2)*45 + theta(3)*85))
fprintf("The student will be admitted\n")

end


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

function grad = gradientCost(x, y, theta, j)

  grad = (htheta(x, theta) - y)*x(j);

end

function h = htheta(x, theta)

  h = sigmoid(sum(theta'.*x));

end

function g = sigmoid(z)

g = 1 ./ (1 + exp(-z));

end
