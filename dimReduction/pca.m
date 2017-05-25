function eigenvalues = pca(x)
    mx = (x - mean(mean(x))) * (x - mean(mean(x)))';
    eigenvalues = sum(mx) / length(mx-1);
end