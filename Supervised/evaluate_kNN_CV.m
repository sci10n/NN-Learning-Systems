%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};

%% Use kNN to classify data
% Note: you have to modify the kNN() function yourselfs.

% Set the number of neighbors
k = 1;

% Cross-validate using the k-fold cv and the number of bins as k
cv_scores = zeros(numBins,1);
n_bin = size(Xt{1},2);
training = zeros(size(Xt{1},1),n_bin * (numBins - 1));
for k_index = 1:8

bin_cv_scores = zeros(numBins,1);
for current_bin = 1:numBins
    
    tmp_index = 0;
    for index = 1:numBins
        if index == current_bin
           continue;
        end
        training(:,(tmp_index*n_bin)+1:((tmp_index+1)*n_bin)) = Xt{tmp_index+1};
        tmp_index = tmp_index +1;
    end
    LkNN = kNN(Xt{current_bin}, k_index, training, Lt{current_bin});
    
    %% Calculate The Confusion Matrix and the Accuracy
    % Note: you have to modify the calcConfusionMatrix() function yourselfs.

    % The confucionMatrix
    cM = calcConfusionMatrix( LkNN, Lt{2});

    % The accuracy
    acc = calcAccuracy(cM);
    bin_cv_scores(current_bin,:) = acc;

end
cv_scores(k_index,:) = sum(bin_cv_scores) / numBins;
end

plot(1:8,cv_scores)
[~,best_score] = max(cv_scores);
