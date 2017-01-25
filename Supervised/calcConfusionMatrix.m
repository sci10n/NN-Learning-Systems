function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
%CALCCONFUSIONMATRIX Summary of this function goes here
%   Detailed explanation goes here

classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

% for i = 1:numClasses
%    for j = 1:numClasses
%        m1 = Lclass == i;
%        m2 = Ltrue == j;
%        mm = m1 == m2;
%       cM(i,j) = sum(mm);
%    end
% end


end

