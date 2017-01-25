function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

classes = unique(Lt);
numClasses = length(classes);
labelsOut  = zeros(size(X,2),1);

% Calculate the distance for all data points in the 
    n = size(X,2);
    
    for index = 1:n
        results = zeros(size(Xt,2),1);
        for index2 = 1:size(Xt,2)
           results(index2,:) =  results(index,:) + sqrt(sum(X(:,index).^2 + Xt(:,index2).^2)); 
        end

    % order results on distance
    results,indexes = sort(results);
    
    %pick the k closest and evaluate label
        % if 50 50 pick the one closest (first in result vector
    closest_labels = zeros(numClases,1);
        for i = 1:k
            class = Lt(indexes(i,:));
            closest_labels(class,:) = closest_labels(class,:) +1
        end
    end
end
