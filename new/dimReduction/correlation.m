function corr = correlation(x,y)
    corr = covariance(x,y) ./ sqrt(variance(x).*variance(y));
end