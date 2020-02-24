clear all;

selectFaces();

trainDirectory = "input/train/";
testingDirectory = "input/test/s";

images = dir(strcat(trainDirectory, "*.pgm"));
imageData = zeros(10304, 320);

for i = 1:size(images)
  currentImage = imread(strcat(trainDirectory, images(i).name));
  imageData(:, i) = currentImage(:);
end

imshow(imageData, [])

saveas(gcf, "output/ps5-1-a.png");

meanVector = mean(imageData, 2);

averageFace = reshape(meanVector, [112, 92]);

imshow(averageFace, [])

saveas(gcf, "output/ps5-1-b.png");

A = zeros(size(imageData));

for i = 1:320
  A(:, i) = imageData(:, i) - meanVector;
end

C = A*A';

imshow(C, []);

saveas(gcf, "output/ps5-1-c.png");

eigen = sort(eig(A'*A), 'DESCEND');

varianceCaptured = zeros(1, 320);

for i = 1:320
  varianceCaptured(i) = v(i, eigen);
end

x = 1:1:320;

plot(x, varianceCaptured);

saveas(gcf, 'output/ps5-1-d.png');

[U, eigenValues] = eigs(C, 162);

imshow(reshape(U(:, 1:8), [112, 92*8]), [])

saveas(gcf, 'output/ps5-1-e.png');

%two parallel arrays, one is w_train, other is the subject corresponding to each row in w_train
training_labels = zeros(320, 1);
w_training = zeros(320, 162);

for i=1:320
  currentImage = imread(strcat(trainDirectory, images(i).name));
  w_training(i, :) = U'*(double(currentImage(:)) - meanVector);
  temp = split(images(i).name,'-');
  training_labels(i) = str2num(string(temp(1)));
end
  %compute w_testing for all stuff in test folder along with parallel labels array testing_labels
  j=1;
  testing_labels = zeros(80, 1);
  w_testing = zeros(80, 162);

for(i=1:40)
  currentDirectory = strcat(testingDirectory, string(i), '/');
  images = dir(strcat(currentDirectory, "*.pgm"));
  currentImage = imread(strcat(currentDirectory, images(1).name));
  testing_labels(j) = i;
  w_testing(j, :) = U'*(double(currentImage(:)) - meanVector);
  j = j+1;
  currentImage = imread(strcat(currentDirectory, images(2).name));
  testing_labels(j) = i;
  w_testing(j, :) = U'*(double(currentImage(:)) - meanVector);
  j=j+1;
end

K = [1, 3, 5, 7, 9, 11];
accuracy_knn = zeros(6, 1);

for i = 1:6
  KNN = fitcknn(w_training, training_labels, 'NumNeighbors', K(i));
  for j=1:80
      [LABEL, POSTERIOR, COST] = predict(KNN, w_testing(j, :));
      if(LABEL == testing_labels(j))
        accuracy_knn(i) = accuracy_knn(i) + 1;
      end
  end
  accuracy_knn(i) = accuracy_knn(i) / 80;
end

T = table(K', accuracy_knn,'VariableNames', {'K_Value' 'Accuracy'});
T

svm_accuracy = zeros(1, 3);

for i=1:40
  svm_training_labels = zeros(320, 1);
  for j=1:8
    svm_training_labels((i-1)*8+j) = i;
  end
  svm_classifier{i} = fitcsvm(w_training, svm_training_labels, 'KernelFunction', 'linear');
end

svm_predictions = zeros(80, 1);

for i=1:80
  best_score = 0;
  for j=1:40
    [LABEL, SCORE] = predict(svm_classifier{j}, w_testing(i, :));
    %%SOMN FUCKED
    if(SCORE(1) > best_score)
      svm_predictions(i) = j;
    end
    %%
  end
  if(testing_labels(i) == svm_predictions(i))
    svm_accuracy(1) = svm_accuracy(1) + 1;
  end
end

svm_accuracy(1) = svm_accuracy(1) / 80;
