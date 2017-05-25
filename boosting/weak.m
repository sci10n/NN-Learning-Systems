function R = weak( X,T,P )

N = size(X,2);
R = zeros(N,1);

for i = 1:N
    if P * X(i) >= P*T
        R(i) = 1;
    else
        R(i) = -1;
    end
end

