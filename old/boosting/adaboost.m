%Load face and non-face data and plot a few examples
load faces, load nonfaces
faces = double(faces); nonfaces = double(nonfaces);

figure(1)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k)), axis image, axis off
end

figure(2)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k)), axis image, axis off
end

% Generate Haar feature masks
nbrHaarFeatures = 200;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:min(nbrHaarFeatures,25)
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
    axis image,axis off
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 400;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

nbrTestExamples = 200;
testImages = cat(3,faces(:,:,nbrTrainExamples:(nbrTrainExamples+nbrTestExamples-1)),nonfaces(:,:,nbrTrainExamples:(nbrTrainExamples+nbrTestExamples-1)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

% Assitance http://mccormickml.com/2013/12/13/adaboost-tutorial/

% Final classifier = sign(sum of all weak classifiers where the ouptu of each classifier is scaled with a weight for that classifier)
% H(X,:) = sign(Sum_{t=1}^{T}alpa(t,:)*output(t))
% alpha(i,:) = 1/2*ln((1-error(i,:))/ error(i,:))
% D(t,i) is a vector of weights, with one weight for eac training example i
% D(t+1,i) = D(t,i) * exp(-alpha(t,:)*y(t,:)*output(t,:)) / sum(sum(D))

M = nbrTrainExamples*2;
accuracy_train = zeros(nbrHaarFeatures,1);
accuracy_test = zeros(nbrHaarFeatures,1);

correct_classifications = zeros(1,nbrTestExamples*2);
for T = 1:nbrHaarFeatures
    d = ones(T,M);
    d(1,:) = 1/M;
    
    polarities = ones(T,1);
    thresholds = zeros(T,1);
    predictions = zeros(T,M);
    alphas = zeros(T,1);
    
    % Calculate the y output for each weak classifier
    for t =1:T
        
        feature_error = 1;
        for i = 1:M
            polarity = 1;
            threshold = xTrain(t,i);
            prediction =  weak(xTrain(t,:), threshold, polarity);
            error = sum( d(t,:) .* sum(yTrain(:) ~= prediction)) / M;
            if error  > 0.5
                error  = 1 - error;
                polarity = -1;
            end
            if error < feature_error
                
                % To fix problem of error = 0 giving infinity
                if error < 0.001
                    feature_error=0.001;
                    
                else
                    feature_error=error;
                    
                end
                thresholds(t,:) = threshold;
                polarities(t,:) = polarity;
                predictions(t,:) = prediction;
            end
        end
        
        alphas(t,1) = 1/2 * log((1-feature_error)/feature_error);
        
        d(t+1,:) = d(t,:) .* exp(-alphas(t,:) * yTrain .* predictions(t,:));
        d(t+1,:) = d(t+1,:) / sum(d(t+1,:));
    end
    
    % Calculate the strong classifier and the result for the test set
    test_predictions = zeros(1,nbrTestExamples*2);
    for i = 1:nbrTestExamples*2
        result = 0;
        for k = 1:T
            result = result + alphas(k,:) .* weak(xTest(k,i),thresholds(k,:),polarities(k,:));
        end
        test_predictions(:,i) = sign(result);
    end
        correct_classifications = correct_classifications + (yTest == test_predictions);

   accuracy_test(T,:) = sum(yTest == test_predictions) / (nbrTestExamples*2);
   
    train_predictions = zeros(1,nbrTrainExamples*2);
    for i = 1:nbrTrainExamples*2
        result = 0;
        for k = 1:T
            result = result + alphas(k,:) .* weak(xTrain(k,i),thresholds(k,:),polarities(k,:));
        end
        train_predictions(:,i) = sign(result);
    end
   accuracy_train(T,:) = sum(yTrain == train_predictions) / (nbrTrainExamples*2);
end

figure(4)
plot(1:nbrHaarFeatures,accuracy_train);

[best_value,best_index] = max(accuracy_test)
figure(5)
plot(1:nbrHaarFeatures,accuracy_test);
hold on;
scatter(best_index,best_value,20,[1,0,0]);

% Get index worst-to-best
[~,I] = sort(correct_classifications(:));

% Get faces / non-faces
faces_index = find(I < nbrTestExamples);
non_faces_index = find(I > nbrTestExamples);

figure(6)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,I(faces_index(k,:)))), axis image, axis off
end

figure(7)
colormap gray
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,I(non_faces_index(k,:)))), axis image, axis off
end