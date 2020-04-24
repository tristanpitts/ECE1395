function temp = fp_Pitts_Tristan_debugging()

clear all;

%read training images
file = fopen('input/train-images-idx3-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numImages = swapbytes(int32(fread(file, 1, 'int32')));

numRows = swapbytes(int32(fread(file, 1, 'int32')));
numCols = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numImages

  temp = fread(file, numRows*numCols, 'uint8');

  trainingImages{i} = reshape(uint8(temp), numRows, numCols)';

end

fclose(file);

%read training labels
file = fopen('input/train-labels-idx1-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numLabels = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numLabels

  trainingLabels(i) = swapbytes(uint8(fread(file, 1, 'uint8')));

end

fclose(file);

%read testing images
file = fopen('input/t10k-images-idx3-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numImages = swapbytes(int32(fread(file, 1, 'int32')));

numRows = swapbytes(int32(fread(file, 1, 'int32')));
numCols = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numImages

  temp = fread(file, numRows*numCols, 'uint8');

  testingImages{i} = reshape(uint8(temp), numRows, numCols)';

end

fclose(file);

%read testing Labels
file = fopen('input/t10k-labels-idx1-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numLabels = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numLabels

  testingLabels(i) = swapbytes(uint8(fread(file, 1, 'uint8')));

end

fclose(file);

indecies = zeros(25, 1);

for i=1:25
  temp = randi(60000, 1);
  while(any(indecies == temp))
    temp = randi(60000, 1);
  end
  indecies(i) = temp;
end

for i=1:25
  subplot(5, 5, i);
  imshow(trainingImages{i});
end

%1a
% This code was used to generate subsets x1-x5
%{
baggingIndecies=randperm(60000);
x1labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x1(i, :) = temp(:);
  x1labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x2labels = zeros(1, 40000clear all;
%{
%read training images
file = fopen('input/train-images-idx3-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numImages = swapbytes(int32(fread(file, 1, 'int32')));

numRows = swapbytes(int32(fread(file, 1, 'int32')));
numCols = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numImages

  temp = fread(file, numRows*numCols, 'uint8');

  trainingImages{i} = reshape(uint8(temp), numRows, numCols)';

end

fclose(file);

%read training labels
file = fopen('input/train-labels-idx1-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numLabels = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numLabels

  trainingLabels(i) = swapbytes(uint8(fread(file, 1, 'uint8')));

end

fclose(file);

%read testing images
file = fopen('input/t10k-images-idx3-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numImages = swapbytes(int32(fread(file, 1, 'int32')));

numRows = swapbytes(int32(fread(file, 1, 'int32')));
numCols = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numImages

  temp = fread(file, numRows*numCols, 'uint8');

  testingImages{i} = reshape(uint8(temp), numRows, numCols)';

end

fclose(file);

%read testing Labels
file = fopen('input/t10k-labels-idx1-ubyte', 'r');

magic=swapbytes(int32(fread(file, 1, 'int32')));

numLabels = swapbytes(int32(fread(file, 1, 'int32')));

for i=1:numLabels

  testingLabels(i) = swapbytes(uint8(fread(file, 1, 'uint8')));

end

fclose(file);

indecies = zeros(25, 1);

for i=1:25
  temp = randi(60000, 1);
  while(any(indecies == temp))
    temp = randi(60000, 1);
  end
  indecies(i) = temp;
end

for i=1:25
  subplot(5, 5, i);
  imshow(trainingImages{i});
end

%1a
% This code was used to generate subsets x1-x5
%{
baggingIndecies=randperm(60000);
x1labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x1(i, :) = temp(:);
  x1labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x2labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x2(i, :) = temp(:);
  x2labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x3labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x3(i, :) = temp(:);
  x3labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x4labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x4(i, :) = temp(:);
  x4labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x5labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x5(i, :) = temp(:);
  x5labels(i) = trainingLabels(baggingIndecies(i));
end

save('input/bagging_variables_and_labels', 'x1', 'x1labels', 'x2', 'x2labels', 'x3', 'x3labels', 'x4',...
 'x4labels', 'x5', 'x5labels');
%}

load('input/bagging_variables_and_labels');

%https://www.mathworks.com/help/stats/fitcecoc.html
%the function fitcecoc uses the one-vs-one approach with support vector machines


%1c
%{
t = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto');
onevonesvm = fitcecoc(double(x1), x1labels', 'Learners', t);
save('input/1c', 'onevonesvm')
%}

load('input/1c')

x1trainingerrorlabels = zeros(size(x1labels));
x1testingerrorlabels = zeros(size(testingLabels));


fprintf("SVM Classifying training set with One vs. One\n");
for i=1:40000
  x1trainingerrorlabels(i) = predict(onevonesvm, double(x1(i, :)));
end

fprintf("SVM Classifying testing with One vs. One\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x1testingerrorlabels(i) = predict(onevonesvm, double(temp(:))');
end

temp = nnz(x1trainingerrorlabels - x1labels);
temp = temp/40000;
fprintf("x1 training error: %0.2f%%\n", temp*100);

temp = nnz(x1testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x1 testing error: %f%%\n", temp*100);



%1d
%{
fprintf("svm 2\n");
t = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto');
onevallsvm = fitcecoc(double(x2), x2labels', 'Learners', t, 'Coding', 'onevsall');
save('input/1d', 'onevsallsvm');
%}

load('input/1d');

x2trainingerrorlabels = zeros(size(x2labels));
x2testingerrorlabels = zeros(size(testingLabels));

fprintf("SVM Classifying training set with One vs. All\n");
for i=1:40000
  x2trainingerrorlabels(i) = predict(onevallsvm, double(x2(i, :)));
end

fprintf("SVM Classifying testing with One vs. All\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x2testingerrorlabels(i) = predict(onevallsvm, double(temp(:))');
end

temp = nnz(x2trainingerrorlabels - x2labels);
temp = temp/40000;
fprintf("x2 training error: %f%%\n", temp*100);

temp = nnz(x2testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x2 testing error: %f%%\n", temp*100);


%1e
%{
x3nnlabels = zeros(10, 40000);

for i=1:40000
  x3nnlabels(x3labels(i)+1, i) = 1;
end


net1 = patternnet(500);
net1 = train(net1, double(x3)', x3nnlabels);
save('input/1e', 'net1');
%}

load('input/1e');

x3trainingerrorlabels = zeros(size(x3labels));
x3testingerrorlabels = zeros(size(testingLabels));

fprintf("Neural Network with 500 hidden units classifying training set\n");
for i=1:40000
  temp = net1(double(x3(i, :))');
  [M, class] = max(temp);
  x3trainingerrorlabels(i) = class - 1;
end

fprintf("Neural Network with 500 hidden units classifying testing set\n");
for i=1:10000
  img = testingImages(i);
  img = cell2mat(img);
  temp = net1(double(img(:)));
  [M, class] = max(temp);
  x3testingerrorlabels(i) = class - 1;
end

temp = nnz(x3trainingerrorlabels - x3labels);
temp = temp/40000;
fprintf("x3 training error: %f%%\n", temp*100);

temp = nnz(x3testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x3 testing error: %f%%\n", temp*100);

%1f
%{
x4nnlabels = zeros(10, 40000);

for i=1:40000
  x4nnlabels(x4labels(i)+1, i) = 1;
end


net2 = patternnet(750);
net2 = train(net2, double(x3)', x3nnlabels);
save('input/1f', 'net2');
%}

load('input/1f');

x4trainingerrorlabels = zeros(size(x4labels));
x4testingerrorlabels = zeros(size(testingLabels));

fprintf("Neural Network with 750 hidden units classifying training set\n");
for i=1:40000
  temp = net2(double(x4(i, :))');
  [M, class] = max(temp);
  x4trainingerrorlabels(i) = class - 1;
end

fprintf("Neural Network with 750 hidden units classifying testing set\n");
for i=1:10000
  img = testingImages(i);
  img = cell2mat(img);
  temp = net2(double(img(:)));
  [M, class] = max(temp);
  x4testingerrorlabels(i) = class - 1;
end

temp = nnz(x4trainingerrorlabels - x4labels);
temp = temp/40000;
fprintf("x4 training error: %f%%\n", temp*100);

temp = nnz(x4testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x4 testing error: %f%%\n", temp*100);

%1g
tree = fitctree(double(x5), x5labels);

x5trainingerrorlabels = zeros(size(x5labels));
x5testingerrorlabels = zeros(size(testingLabels));

fprintf("Decision tree classifying training set\n");
for i=1:40000
  x5trainingerrorlabels(i) = predict(tree, double(x5(i, :)));
end

fprintf("Decision tree classifying testing set\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x5testingerrorlabels(i) = predict(tree, double(temp(:))');
end

temp = nnz(x5trainingerrorlabels - x5labels);
temp = temp/40000;
fprintf("x5 training error: %f%%\n", temp*100);

temp = nnz(x5testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x5 testing error: %f%%\n", temp*100);


%1h
fprintf("Classifying testing set with majority voting rule\n");
majorityLabels = zeros(10000, 1);
for i=1:10000
  temp = mode([x1testingerrorlabels(i), x2testingerrorlabels(i), x3testingerrorlabels(i), x4testingerrorlabels(i), x5testingerrorlabels(i)]);
  majorityLabels(i) = temp;
end

majorityError = nnz(majorityLabels' - double(testingLabels))/10000;
fprintf("Majoity Voting Error: %f%%\n", majorityError*100);
close;


%problem 2
img1 = imread('input/im1.jpg');
h1 = size(img1, 1);
w1 = size(img1, 2);
img2 = imread('input/im2.jpg');
h2 = size(img2, 1);
w2 = size(img2, 2);
img3 = imread('input/im3.png');
h3 = size(img3, 1);
w3 = size(img3, 2);

img1 = imresize(im2double(img1), [100, 100]);
img2 = imresize(im2double(img2), [100, 100]);
img3 = imresize(im2double(img3), [100, 100]);


img1 = reshape(img1, 10000, 3);
img2 = reshape(img2, 10000, 3);
img3 = reshape(img3, 10000, 3);

%test = reshape(img1, 100, 100, 3);
%imshow(test);

K=[2, 3, 5, 7];
iters = [7, 13, 20];
R = [5, 15, 25];

fprintf("Running KMeans on image 1\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img1, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img1-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end

fprintf("Running KMeans on image 2\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img2, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img2-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end

fprintf("Running KMeans on image 3\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img3, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img3-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end
);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x2(i, :) = temp(:);
  x2labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x3labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x3(i, :) = temp(:);
  x3labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x4labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x4(i, :) = temp(:);
  x4labels(i) = trainingLabels(baggingIndecies(i));
end

baggingIndecies=randperm(60000);
x5labels = zeros(1, 40000);

for i=1:40000
  temp = trainingImages{baggingIndecies(i)};
  x5(i, :) = temp(:);
  x5labels(i) = trainingLabels(baggingIndecies(i));
end

save('input/bagging_variables_and_labels', 'x1', 'x1labels', 'x2', 'x2labels', 'x3', 'x3labels', 'x4',...
 'x4labels', 'x5', 'x5labels');
%}

load('input/bagging_variables_and_labels');

%https://www.mathworks.com/help/stats/fitcecoc.html
%the function fitcecoc uses the one-vs-one approach with support vector machines


%1c
%{
t = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto');
onevonesvm = fitcecoc(double(x1), x1labels', 'Learners', t);
save('input/1c', 'onevonesvm')
%}

load('input/1c')

x1trainingerrorlabels = zeros(size(x1labels));
x1testingerrorlabels = zeros(size(testingLabels));


fprintf("SVM Classifying training set with One vs. One\n");
for i=1:40000
  x1trainingerrorlabels(i) = predict(onevonesvm, double(x1(i, :)));
end

fprintf("SVM Classifying testing with One vs. One\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x1testingerrorlabels(i) = predict(onevonesvm, double(temp(:))');
end

temp = nnz(x1trainingerrorlabels - x1labels);
temp = temp/40000;
fprintf("x1 training error: %0.2f%%\n", temp*100);

temp = nnz(x1testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x1 testing error: %f%%\n", temp*100);



%1d
%{
fprintf("svm 2\n");
t = templateSVM('KernelFunction','rbf', 'KernelScale', 'auto');
onevallsvm = fitcecoc(double(x2), x2labels', 'Learners', t, 'Coding', 'onevsall');
save('input/1d', 'onevsallsvm');
%}

load('input/1d');

x2trainingerrorlabels = zeros(size(x2labels));
x2testingerrorlabels = zeros(size(testingLabels));

fprintf("SVM Classifying training set with One vs. All\n");
for i=1:40000
  x2trainingerrorlabels(i) = predict(onevallsvm, double(x2(i, :)));
end

fprintf("SVM Classifying testing with One vs. All\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x2testingerrorlabels(i) = predict(onevallsvm, double(temp(:))');
end

temp = nnz(x2trainingerrorlabels - x2labels);
temp = temp/40000;
fprintf("x2 training error: %f%%\n", temp*100);

temp = nnz(x2testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x2 testing error: %f%%\n", temp*100);


%1e
%{
x3nnlabels = zeros(10, 40000);

for i=1:40000
  x3nnlabels(x3labels(i)+1, i) = 1;
end


net1 = patternnet(500);
net1 = train(net1, double(x3)', x3nnlabels);
save('input/1e', 'net1');
%}

load('input/1e');

x3trainingerrorlabels = zeros(size(x3labels));
x3testingerrorlabels = zeros(size(testingLabels));

fprintf("Neural Network with 500 hidden units classifying training set\n");
for i=1:40000
  temp = net1(double(x3(i, :))');
  [M, class] = max(temp);
  x3trainingerrorlabels(i) = class - 1;
end

fprintf("Neural Network with 500 hidden units classifying testing set\n");
for i=1:10000
  img = testingImages(i);
  img = cell2mat(img);
  temp = net1(double(img(:)));
  [M, class] = max(temp);
  x3testingerrorlabels(i) = class - 1;
end

temp = nnz(x3trainingerrorlabels - x3labels);
temp = temp/40000;
fprintf("x3 training error: %f%%\n", temp*100);

temp = nnz(x3testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x3 testing error: %f%%\n", temp*100);

%1f
%{
x4nnlabels = zeros(10, 40000);

for i=1:40000
  x4nnlabels(x4labels(i)+1, i) = 1;
end


net2 = patternnet(750);
net2 = train(net2, double(x3)', x3nnlabels);
save('input/1f', 'net2');
%}

load('input/1f');

x4trainingerrorlabels = zeros(size(x4labels));
x4testingerrorlabels = zeros(size(testingLabels));

fprintf("Neural Network with 750 hidden units classifying training set\n");
for i=1:40000
  temp = net2(double(x4(i, :))');
  [M, class] = max(temp);
  x4trainingerrorlabels(i) = class - 1;
end

fprintf("Neural Network with 750 hidden units classifying testing set\n");
for i=1:10000
  img = testingImages(i);
  img = cell2mat(img);
  temp = net2(double(img(:)));
  [M, class] = max(temp);
  x4testingerrorlabels(i) = class - 1;
end

temp = nnz(x4trainingerrorlabels - x4labels);
temp = temp/40000;
fprintf("x4 training error: %f%%\n", temp*100);

temp = nnz(x4testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x4 testing error: %f%%\n", temp*100);

%1g
tree = fitctree(double(x5), x5labels);

x5trainingerrorlabels = zeros(size(x5labels));
x5testingerrorlabels = zeros(size(testingLabels));

fprintf("Decision tree classifying training set\n");
for i=1:40000
  x5trainingerrorlabels(i) = predict(tree, double(x5(i, :)));
end

fprintf("Decision tree classifying testing set\n");
for(i=1:10000)
  temp = testingImages(i);
  temp = cell2mat(temp);
  x5testingerrorlabels(i) = predict(tree, double(temp(:))');
end

temp = nnz(x5trainingerrorlabels - x5labels);
temp = temp/40000;
fprintf("x5 training error: %f%%\n", temp*100);

temp = nnz(x5testingerrorlabels - double(testingLabels));
temp = temp/10000;
fprintf("x5 testing error: %f%%\n", temp*100);


%1h
fprintf("Classifying testing set with majority voting rule\n");
majorityLabels = zeros(10000, 1);
for i=1:10000
  temp = mode([x1testingerrorlabels(i), x2testingerrorlabels(i), x3testingerrorlabels(i), x4testingerrorlabels(i), x5testingerrorlabels(i)]);
  majorityLabels(i) = temp;
end

majorityError = nnz(majorityLabels' - double(testingLabels))/10000;
fprintf("Majoity Voting Error: %f%%\n", majorityError*100);
close;
%}

%problem 2
img1 = imread('input/im1.jpg');
h1 = size(img1, 1);
w1 = size(img1, 2);
img2 = imread('input/im2.jpg');
h2 = size(img2, 1);
w2 = size(img2, 2);
img3 = imread('input/im3.png');
h3 = size(img3, 1);
w3 = size(img3, 2);

img1 = imresize(im2double(img1), [100, 100]);
img2 = imresize(im2double(img2), [100, 100]);
img3 = imresize(im2double(img3), [100, 100]);


img1 = reshape(img1, 10000, 3);
img2 = reshape(img2, 10000, 3);
img3 = reshape(img3, 10000, 3);

%test = reshape(img1, 100, 100, 3);
%imshow(test);

K=[2, 3, 5, 7];
iters = [7, 13, 20];
R = [5, 15, 25];

fprintf("Running KMeans on image 1\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img1, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img1-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end

fprintf("Running KMeans on image 2\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img2, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img2-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end

fprintf("Running KMeans on image 3\n");
for i=1:4
  for j=1:3
    for k=1:3
      [ids, means, ssd] = kmeans_multiple(img3, K(i), iters(j), R(k));
      for(m = 1:10000)
        newImg(m, :) = means(ids(m), :);
      end
      newImg = im2uint8(reshape(newImg, 100, 100, 3));
      imshow(newImg);
      filename = sprintf("output/fp-2-c-img3-%d-%d-%d.png", K(i), iters(j), R(k));
      saveas(gcf, filename);
      clear newImg;
    end
  end
end

end
