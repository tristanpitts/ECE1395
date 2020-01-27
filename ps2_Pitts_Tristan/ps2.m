clear all;

%question 4
%4a
A = csvread('input/hw2_data1.txt');

X = A(:, 1);
Y = A(:, 2);

%4b
scatter(X,Y,'red','X');
xlabel("Population")
ylabel("Profit")

%set dummy variable x1 in every row to 1
X = [ones(size(X,1),1), X(:,1)];

%4c
fprintf("The size of the feature matrix X is [%d, %d] and the size of the label vector y is [%d, %d]\n", size(X), size(Y));

%4d
fprintf("The cost associated with theta = [0; 0] is %f\n", computeCost(X, Y, [0;0]))

%4e
[theta, cost] = gradientDescent(X, Y, 0.01, 1500);

x_approx = X(:, 2);
y_approx = theta(1) + theta(2)*x_approx;

hold on
plot(x_approx, y_approx)
hold off
iteration = 1:1:1500;

plot(iteration, cost)
xlabel("Iteration")
ylabel("Cost")
fprintf("The computed model parameters are: theta1 = %f and theta2 = %f\n", theta(1), theta(2))

%4f
fprintf("using the obtained model parameters, the profit in a city of population 35000 is estimated to be %f\n", theta(1) + theta(2)*35000)
fprintf("using the obtained model parameters, the profit in a city of population 70000 is estimated to be %f\n", theta(1) + theta(2)*70000)

%4g
thetan = normalEqn(X, Y);

fprintf("using the normalEqn function, the profit in a city of population 35000 is estimated to be %f\n", thetan(1) + thetan(2)*35000)
fprintf("using the normalEqn function, the profit in a city of population 70000 is estimated to be %f\n", thetan(1) + thetan(2)*70000)

%4h
iteration = 1:1:250;
[theta, cost] = gradientDescent(X, Y, 0.0001, 250);
plot(iteration, cost)
hold on
[theta, cost] = gradientDescent(X, Y, 0.001, 250);
plot(iteration, cost)
[theta, cost] = gradientDescent(X, Y, 0.03, 250);
plot(iteration, cost)
[theta, cost] = gradientDescent(X, Y, 0.1, 250);
plot(iteration, cost)
[theta, cost] = gradientDescent(X, Y, 1, 250);
plot(iteration, cost)
legend("alpha = 0.0001", "alpha = 0.001", "alpha = 0.03", "alpha = 0.1", "alpha = 1");
axis([0 10 0 100])
xlabel("Iteration")
ylabel("Cost")
hold off


%question 5
%5a
A = csvread('input/hw2_data2.txt');
X = [A(:, 1), A(:,2)];
Y = A(:, 3);

fprintf("The size of the feature matrix X is %d and the size of the label vector Y is %d\n", size(X), size(Y))

stdevx1 = std(X(:,1));
meanx1 = mean(X(:,1));
X(:,1) = (X(:,1) - meanx1) / stdevx1;
fprintf("The mean and standard deviation of x1 are %f and %f\n", meanx1, stdevx1)

stdevx2 = std(X(:,2));
meanx2 = mean(X(:,2));
X(:,2) = (X(:,2) - meanx2) / stdevx2;
fprintf("The mean and standard deviation of x2 are %f and %f\n", meanx2, stdevx2)

X = [ones(size(X,1),1), X(:,1), X(:,2)];

%5b
[theta, cost] = gradientDescent(X, Y, 0.01, 1500);
iteration = 1:1:1500;
plot(iteration, cost)
xlabel("Iteration")
ylabel("Cost")
fprintf("The computed model parameters are: theta1 = %f, theta2 = %f, theta3 = %f\n", theta(1), theta(2), theta(3))

%5c
x1 = (1650-meanx1)/stdevx1;
x2 = (3-meanx2)/stdevx2;
fprintf("Based on the model parameters obtained in question 5B, the predicted cost of a 1650 square foot house with 3 bedrooms is %f\n", theta(1) + theta(2)*x1 + theta(3)*x2)
