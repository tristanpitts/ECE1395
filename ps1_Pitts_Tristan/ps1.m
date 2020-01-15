clear all;

%Section 3
%3a
x = 0 + 3.7*randn(1000000, 1);

%3b
fprintf("The min element of a is %f\n", min(x))
fprintf("The max element of a is %f\n", max(x))
fprintf("The mean of the elements in a is %f\n", mean(x))
fprintf("The standard deviation of the elements in a is %f\n", std(x))

%3c
xnorm = normalize(x);
histogram(xnorm)
saveas(gcf,sprintf("output/ps1-3-c.png",i));

%3d
t1 = cputime;
for i=1:size(xnorm)
    xnorm(i)=xnorm(i)+1;
end
fprintf("Elapsed Time: %f\n", cputime - t1)

%3e
t2 = cputime;
x = x + 1;
cast(cputime - t2, 'double')

%3f
xx = cast(x, 'int32');
y = [];
for i=1:size(xx)
    if mod(xx(i),2) == 0 && xx(i) > 0 && xx(i) < 100 %if xx(i) is even and between 0 and 100
        y(size(y)+1)=xx(i);
    end
end

figure(); %need to call figure to be able to display multiple plots
hist(y);
saveas(gcf, sprintf("output/ps1-3-e.png",i))

%Section 4

%4a
A = [2,1,3;2,6,8;6,8,18]
fprintf("The minimum value in the first row of A is %d\n", min( A(1,:)))
fprintf("The minimum value in the second row of A is %d\n", min( A(2,:)))
fprintf("The minimum value in the third row of A is %d\n", min( A(3,:)))
fprintf("The maximum value in the first column of A is %d\n", max(A(:,1)))
fprintf("The maximum value in the second column of A is %d\n", max(A(:,2)))
fprintf("The maximum value in the third column of A is %d\n", max(A(:,3)))
fprintf("The largest value in A is %d\n", max(max(A)))
B = A.^2

%4b
C = [2 1 3;2 6 8;6 8 18] %coeffecients of x,y and z
D = [1;3;5]

F = inv(C)*D %resulting vector  will give x, y and z

%4c
x1=[0.5 0 1.5];
x2=[1 1 0];
fprintf("The L1 norm of x1 (the sum of all elements in x1) is: %d\n", sum(x1))
fprintf("The L2 norm of x1 (the square root of the sum of the squares of all elements in x1) is: %f\n", (x1(1)^2 + x1(2)^2 + x1(3)^2)^(0.5))
fprintf("Matlab's norm of x1: %f\n\n", norm(x1))
fprintf("The L1 norm of x2 (the sum of all elements in x2) is: %d\n", sum(x2))
fprintf("The L2 norm of x2 (the square root of the sum of the squares of all elements in x2) is: %f\n", (x2(1)^2 + x2(2)^2 + x2(3)^2)^(0.5))
fprintf("Matlab's norm of x2: %f\n\n", norm(x2))

%Section 5
A = [1 2 3; 4 5 6; 7 8 9]
normalize_rows(A);

A = [10 17 21; 4 2 99; 12 144 3; 4 10 13; 1 2 44]
normalize_rows(A)
