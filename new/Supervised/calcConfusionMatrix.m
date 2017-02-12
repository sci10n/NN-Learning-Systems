function [ cM ] = calcConfusionMatrix( Lclass, Ltrue )
classes = unique(Ltrue);
numClasses = length(classes);
cM = zeros(numClasses);

 for index = 1:length(Ltrue)
    true_label = Ltrue(index);
    observed_label = Lclass(index);
    cM(observed_label,true_label) = cM(observed_label,true_label) + 1;

 end

end

