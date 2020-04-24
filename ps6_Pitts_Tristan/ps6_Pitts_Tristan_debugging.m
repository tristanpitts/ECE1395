clear all;

data = csvread('input/diabetes.csv', 1);

fid = fopen('input/diabetes.csv', 'r');
labels = split(fgetl(fid), ',');
fclose(fid);

indecies = zeros(540,1);
x_train = zeros(540,8);
x_test = zeros(228,8);
y_train = zeros(540,1);
y_test = zeros(228,1);


for(i = 1:540)
  temp = randi([1, 768], 1);
  while(any(indecies==temp))
    temp = randi([1, 768], 1);
  end
  indecies(i) = temp;
end
traincount = 1;
testcount = 1;

for(i = 1:size(data, 1))
  if(any(indecies==i))
    x_train(traincount, :) = data(i, 1:8);
    y_train(traincount) = data(i, 9);
    traincount = traincount + 1;
  else
    x_test(testcount, :) = data(i, 1:8);
    y_test(testcount) = data(i, 9);
    testcount = testcount + 1;
  end
end

%1a
x_train_0 = zeros(1, 8);
x_train_1 = zeros(1, 8);
count0 = 1;
count1 = 1;


for(i=1:size(x_train, 1))
  if(y_train(i) == 1)
    x_train_1(count1, :) = x_train(i, :);
    count1 = count1 + 1;
  else
    x_train_0(count0, :) = x_train(i, :);
    count0 = count0 + 1;
  end
end

%1b
class_0_means = mean(x_train_0);
class_0_stdevs = std(x_train_0);
class_1_means = mean(x_train_1);
class_1_stdevs = std(x_train_1);

t = table(class_0_means', class_1_means', class_0_stdevs', class_1_stdevs', 'VariableNames', {'Class_0_Means', 'Class_1_Means', 'Class_0_STDEV', 'Class_1_STDEV'});
t.variables = labels(1:8);
t

%1c
testingProb0 = zeros(size(x_test, 1), 8);
testingProb1 = zeros(size(x_test, 1), 8);


for(i=1:size(x_test, 1))
  for(j=1:8)
    testingProb0(i, j) = sqrt(2*pi*class_0_stdevs(j))*exp(-1*((x_test(i, j)-class_0_means(j))^2)/(2*class_0_stdevs(j)));
    testingProb1(i, j) = sqrt(2*pi*class_1_stdevs(j))*exp(-1*((x_test(i, j)-class_1_means(j))^2)/(2*class_1_stdevs(j)));
  end
end

ClassProbabilities = ones(size(x_test, 1), 2);

ClassProbabilities(:, 1) = prod(testingProb0, 2);
ClassProbabilities(:, 2) = prod(testingProb1, 2);

posteriorProb = zeros(size(ClassProbabilities));

posteriorProb(:, 1) = ClassProbabilities(:, 1) * 0.65;
posteriorProb(:, 2) = ClassProbabilities(:, 2) * 0.35;

class = zeros(size(posteriorProb, 1), 1);

for(i=1:size(posteriorProb, 1))
  if(posteriorProb(i, 1)) > posteriorProb(i, 2)
    class(i) = 0;
  else
    class(i) = 1;
  end
end

naiveAccuracy = (size(y_test) - sum(abs(y_test - class))) / size(y_test);

%QUESTION 2

C = cov(x_train);
C
mahalanobisClass = zeros(size(x_test, 1), 1);

for(i=1:size(x_test, 1));
  d0 = ((x_test(i, :)-class_0_means) * (C.^-1) * (x_test(i, :)-class_0_means)')^(0.5);
  d1 = ((x_test(i, :)-class_1_means) * (C.^-1) * (x_test(i, :)-class_1_means)')^(0.5);

  dprime0 = d0 - log(0.65);
  dprime1 = d1 - log(0.35);

  if(dprime0 < dprime1)
    mahalanobisClass(i) = 0;
  else
    mahalanobisClass(i) = 1;
  end
end

mahalanobisAccuracy = (size(y_test) - sum(abs(y_test - mahalanobisClass))) / size(y_test);
