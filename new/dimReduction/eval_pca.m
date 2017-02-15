
% Print covaraince matrix
n = size(countrydata,1);
covariance_matrix = zeros(n);
for i = 1:n
    for j = 1:n
        covariance_matrix(i,j) = covariance(countrydata(i,:),countrydata(j,:));
    end
end
figure(1101)
clf
image(covariance_matrix);
colorbar
% Print correlation matrix, invariant to scaling because of the normalizing
correlation_matrix = zeros(n);
for i = 1:n
    for j = 1:n
        correlation_matrix(i,j) = correlation(countrydata(i,:),countrydata(j,:));
    end
end
figure(1102)
clf
imagesc(correlation_matrix);
colorbar


% function sqrt(Var(X)*Var(y))
% Normalize each of the features
normed_data = normalize(countrydata);

eigenvalues = pca(normed_data);

figure(1103)
clf
hist(eigenvalues)

[sorted,sorted_index]= sort(eigenvalues,2,'descend')

figure(1104)
clf
scatter(normed_data(sorted_index(1),:),normed_data(sorted_index(2),:), 20, countryclass)
colorbar

s1 = countrydata(:,countryclass == 0);
s2 = countrydata(:,countryclass == 2);
w = fld(s1,s2)

