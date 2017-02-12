% Load face and non-face data and plot a few examples
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
nbrHaarFeatures = 400;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:min(nbrHaarFeatures,25)
subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
axis image,axis off
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 200;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

% Assitance http://mccormickml.com/2013/12/13/adaboost-tutorial/

% Final classifier = sign(sum of all weak classifiers where the ouptu of each classifier is scaled with a weight for that classifier)
% H(X,:) = sign(Sum_{t=1}^{T}alpa(t,:)*output(t))
% alpha(i,:) = 1/2*ln((1-error(i,:))/ error(i,:))
% D(t,i) is a vector of weights, with one weight for eac training example i
% D(t+1,i) = D(t,i) * exp(-alpha(t,:)*y(t,:)*output(t,:)) / sum(sum(D))

% Calculate the y output for each weak classifier
M = nbrTrainExamples*2;
accuracy = zeros(nbrHaarFeatures,1);
for T = 1:nbrHaarFeatures
d = ones(T,M);
d(1,:) = 1/M;

alphas = zeros(T,1);
results = zeros(T,M);
for t =1:T
   
    error = sum(d(t,:) .* (yTrain ~= sign(xTrain(t,:))));
   
   alphas(t,:) = 1/2 * log((1-error)/error);
  
   d(t+1,:) = d(t,:) .* exp(-alphas(t,:) * yTrain .* sign(xTrain(t,:)));
   d(t+1,:) = d(t+1,:) / sum(d(t+1,:));
  
   results(t,:) =  sign(xTrain(t,:));

end

a = repmat(alphas,1,M);
strong = sign(sum(a .* results));

accuracy(T,:) = sum(strong == yTrain)/M;
end

figure(4)
plot(1:nbrHaarFeatures,accuracy);
% nbrHaarFeatures = 100;
% haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
% repmat(alphas,[T,M])
% nbrTestExamples = 500;
% testImages = cat(3,faces(:,:,1:nbrTestExamples),nonfaces(:,:,1:nbrTestExamples));
% xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
% yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];
% strong = sign(sum(alphas.*sign(xTest)));
% sum(yTest == strong)/M
