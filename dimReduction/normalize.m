function y = normalize(x)
    y = x - mean(mean(x));
    y = y ./ repmat(sqrt(sum(y.^2,2)),1,size(y,2));
    
end
