function [ acc ] = calcAccuracy( cM )
%CALCACCURACY Takes a confusion matrix amd calculates the accuracy

acc = 0;
for index = 1:size(cM,1)
   acc = acc + cM(index,index); 
end
acc = acc / sum(sum(cM));

end

