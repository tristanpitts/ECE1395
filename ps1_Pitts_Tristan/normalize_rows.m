function [B] = normalize_rows(A)
    B = A ./ repmat(sum(A,ndims(A)),1,size(A,2))
    %Assuming we have an MxN matrix, the above statement computes the sum of each row in the matrix and
    %returns that as a 1xN vector. It then copies that vector as many times
    %as neccessary to form an MxN matrix where each column is the 1xN
    %vector we obtained. Finally, element-by-element division is performed
    %such that each element of the original matrix is divided by the sum of
    %the elements in the row that it corresponds to. This leaves us with an
    %array where the sum of all rows is 1.
end