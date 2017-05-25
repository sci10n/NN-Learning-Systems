function var = variance(x)
    mx = (x - mean(mean(x))).^2;
    var = sum(mx,2) / length(x);
end