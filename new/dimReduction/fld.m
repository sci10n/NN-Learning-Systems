function w = fld(x,y)
    mx = mean(mean(x));
    my = mean(mean(y));
    w = (1/(covariance(x,x) + covariance(y,y))) * (my - mx);
end