function cov = covariance(x,y)
    mx = x - mean(mean(x));
    my = y - mean(mean(y));
    cov = sum(mx .* my,2);
    cov = cov / (length(mx)-1);
end