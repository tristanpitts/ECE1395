clear all;

data = csvread('input/diabetes.csv', 1);

fid = fopen('input/diabetes.csv', 'r');
labels = split(fgetl(fid), ',');
fclose(fid);

training_set_size = [230, 384, 537, 691];
h = [0.1, 0.4, 0.7, 1.0, 1.5];
accuracies = zeros(4, 5);

for k=1:4
  for d = 1:5
    %Divide data into testing and training
    indecies = zeros(training_set_size(k),1);
    x_train = zeros(training_set_size(k),8);
    x_test = zeros(768-training_set_size(k),8);
    y_train = zeros(training_set_size(k),1);
    y_test = zeros(768-training_set_size(k),1);


    for(i = 1:training_set_size(k))
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

    predictions = zeros(size(x_test, 1), 1);

    for i=1:size(x_test, 1)
      temp = zeros(2,1);

      for j=1:size(x_train_0, 1)
        temp(1) = temp(1) + (1 / (((2*pi)^4)*h(d)^8))*exp(-(x_test(i, :)-(x_train_0(j, :)))*(x_test(i, :)-(x_train_0(j, :)))' / 2*h(d)^2);
        temp(1) = temp(1) / size(x_train_0, 1);
      end

      for j=1:size(x_train_1)
        temp(2) = temp(2) + (1 / (((2*pi)^4)*h(d)^8))*exp(-(x_test(i, :)-(x_train_1(j, :)))*(x_test(i, :)-(x_train_1(j, :)))' / 2*h(d)^2);
        temp(2) = temp(2) / size(x_train_1, 1);
      end

      if (temp(1)*0.65 > temp(2)*.35)
        predictions(i) = 0;
      else
        predictions(i) = 1;
      end
    end
    accuracies(k, d) = ((768 - training_set_size(k)) - nnz(y_test - predictions))/(768 - training_set_size(k));
  end
end

rows = {'N=230', 'N=384', 'N=537', 'N=691'};
columns = {'h0_1','h0_4','h0_7','h1_0', 'h1_5'};

T = array2table(accuracies, 'RowNames', rows, 'VariableNames', columns);

T
