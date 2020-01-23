clear all;

A = csvread('input/hw2_data1.txt');

X = A(:, 1);
Y = A(:, 2);

scatter(X,Y,'red','X');

X = [ones(size(X,1),1), X(:,1)];

computeCost(X, Y, [0;0])

[theta, cost] = gradientDescent(X, Y, 0.01, 1500);

x_approx = X(:, 2);
y_approx = theta(1) + theta(2)*x_approx;

hold on
plot(x_approx, y_approx)
hold off

A = csvread('input/hw2_data2.txt');

X = [A(:, 1), A(:,2)];
X = [ones(size(X,1),1), X(:,1), X(:,2)];
Y = A(:, 3);

[theta, cost] = gradientDescent(X, Y, 0.01, 1500);
cost
