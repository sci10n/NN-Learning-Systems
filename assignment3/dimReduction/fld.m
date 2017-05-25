function w = fld(x,y)

w = [];
n = size(x,1);
for i = 1:n

    xt = x(i,:);
    yt = y(i,:);
        mx = mean(xt);
    my = mean(yt);
    w(i,:) = (1/(covariance(xt,xt) + covariance(yt,yt))) * (my - mx);
end
end