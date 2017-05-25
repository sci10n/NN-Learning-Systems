
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
[sorted,sorted_index]= sort(eigenvalues,2,'descend')

%[A,B] = sorteig(normed_data);
figure(1103)
clf
bar(sorted)


figure(1104)
clf
scatter(countrydata(sorted_index(1),:) * sorted(:,1),countrydata(sorted_index(2),:)* sorted(:,2), 20, countryclass,'filled')
hold on;
% Plot Georgia
scatter(countrydata(sorted_index(1),41)* sorted(:,1),countrydata(sorted_index(2),41)* sorted(:,2), 20, [1 0 0],'filled')

% Find outlier
[~,A] = max(countrydata(sorted_index(1),:) * sorted(:,1));
countries(A,:)

s1 = countrydata(:,countryclass == 0);
s2 = countrydata(:,countryclass == 2);
w = fld(s1,s2)

figure(1105)
clf
plot(s1'*w, 'xb');
hold on;
plot(s2'*w,'or');



