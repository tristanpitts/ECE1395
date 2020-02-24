function selectFaces()

%generate two different integers between 1 and 10, these photos are used for testing, the others are used for training
%string(1) returns 1 as a string
%copyfile source destination
sourceDirectory = "input/all/s";
testDirectory = "input/test/s";
trainDirectory = "input/train/";

delete(strcat(trainDirectory, '*'));
for i = 1:40
  test1 = randi(10);
  test2 = randi(10);
  while(test1 == test2)
    test2 = randi(10);
  end
  currentDirectory = strcat(sourceDirectory, string(i), '/');
  for j = 1:10
    currentTestDir = strcat(testDirectory, string(i), '/');
    if(j == 1)
      delete(strcat(currentTestDir, '*'));
    end
    currentFile = strcat(currentDirectory, string(j), ".pgm");
    if(j == test1 || j == test2)
        copyfile(currentFile, currentTestDir);
    else
        copyfile(currentFile, strcat(trainDirectory, string(i), '-', string(j), ".pgm"));
    end
end

end
