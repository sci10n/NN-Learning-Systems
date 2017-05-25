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

labelsOut  = zeros(size(X,2),1);
classes = unique(Lt);
numClasses = length(classes);


% Calculate the distance for all data points in the 
    n = size(X,2);
    
    for index = 1:n
        results = zeros(size(Xt,2),1);
        for index2 = 1:size(Xt,2)
            % Value for the point we want to classify
            v1 = X(:,index);
            % Value for the point in the training set
            v2 = Xt(:,index2);
           results(index2,:) = sqrt(sum((v1 - v2).^2)); 
        end
    % order results on distance
    [~,I] = sort(results);
    indexes = I;
    % pick the k closest and evaluate label
        % if 50 50 pick the one closest (first in result vector
    count_labels = zeros(numClasses,1);
    
        for i = 1:k
            class = Lt(indexes(i,:));
            count_labels(class,:) = count_labels(class,:) +1;
        end
        [~,I] = max(count_labels);
        labelsOut(index,:) = I;
    end

end

