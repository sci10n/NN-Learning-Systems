%% This script will help you test out your single layer neural network code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training features

numBins = 2; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = Inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = false; % true = select features at random, false = select the first features

[ Xt, Dt, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: Xt, Dt, Lt will be cell arrays, to extract a bin from them use i.e.
% XBin1 = Xt{1};


%% Modify the X Matrices so that a bias is added
% The Training Data
Xtraining = Xt{1};

% The Test Data
Xtest = Xt{2};

% Normalize features

% mean = mean(X,2);
% dev = std(X,[],2);
% 
% for i = 1:size(Xtraining, 1)
%         Xtraining(i,:) = (Xtraining(i,:)-mean(i))*(1/dev(i));
% end
% for i = 1:size(Xtest, 1)
%         Xtest(i,:) = (Xtest(i,:)-mean(i))*(1/dev(i));
% end
Xtraining = [Xtraining;ones(1,size(Xtraining,2))];
Xtest = [Xtest;ones(1,size(Xtest,2))];

%% Train your single layer network
% Note: You nned to modify trainSingleLayer() in order to train the network

numHidden = 10; % Change this, Number of hidde neurons 
numIterations = 10000; % Change this, Numner of iterations (Epochs)
learningRate = 0.02; % Change this, Your learningrate
W0 = (rand(numHidden+1,size(X,1)+1) -0.5)* 0.05; % Change this, Initiate your weight matrix W

V0 = (rand(size(D,1),numHidden+1) -0.5)* 0.05;  % Change this, Initiate your weight matrix V
testSamples = 999;%size(Xtraining,2);
%
tic
[W,V, trainingError, testError ] = trainMultiLayer(Xtraining(:,1:testSamples),Dt{1}(:,1:testSamples),Xtest(:,1:testSamples),Dt{2}(:,1:testSamples), W0,V0,numIterations, learningRate );
trainingTime = toc;
%% Plot errors
figure(1101)
clf
[mErr, mErrInd] = min(testError);
plot(trainingError,'k','linewidth',1.5)
hold on
plot(testError,'r','linewidth',1.5)
plot(mErrInd,mErr,'bo','linewidth',1.5)
hold off
title('Training and Test Errors, Multi-Layer')
legend('Training Error','Test Error','Min Test Error')

%% Calculate The Confusion Matrix and the Accuracy of the Evaluation Data
% Note: you have to modify the calcConfusionMatrix() function yourselfs.

[ Y, LMultiLayerTraining ] = runMultiLayer(Xtraining, W, V);
tic
[ Y, LMultiLayerTest ] = runMultiLayer(Xtest, W,V);
classificationTime = toc/length(Xtest);
% The confucionMatrix
cM = calcConfusionMatrix( LMultiLayerTest, Lt{2});

% The accuracy
acc = calcAccuracy(cM);

display(['Time spent training: ' num2str(trainingTime) ' sec'])
display(['Time spent calssifying 1 feature vector: ' num2str(classificationTime) ' sec'])
display(['Accuracy: ' num2str(acc)])

%% Plot classifications
% Note: You do not need to change this code.

if dataSetNr < 4
    plotResultMultiLayer(W,V,Xtraining,Lt{1},LMultiLayerTraining,Xtest,Lt{2},LMultiLayerTest)
else
    plotResultsOCR( Xtest, Lt{2}, LMultiLayerTest )
end
