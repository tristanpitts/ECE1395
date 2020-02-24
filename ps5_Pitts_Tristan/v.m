function v = v(k, eigen)

numerator = 0;
denominator = sum(eigen, 'all');

for i = 1:k
   numerator = numerator + eigen(i, 1);
end

v = numerator/denominator;

end
