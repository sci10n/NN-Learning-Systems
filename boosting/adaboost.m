\% Load face and non-face data and plot a few examples
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

\% Generate Haar feature masks
nbrHaarFeatures = ?;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
figure(3)
colormap gray
for k = 1:25
subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2])
axis image,axis off
end

\% Create a training data set with a number of training data examples
\% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = ?;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

% Assitance http://mccormickml.com/2013/12/13/adaboost-tutorial/

% Final classifier = sign(sum of all weak classifiers where the ouptu of each classifier is scaled with a weight for that classifier)
% H(X,:) = sign(Sum_{t=1}^{T}alpa(t,:)*output(t))
% alpha(i,:) = 1/2*ln((1-error(i,:))/ error(i,:))
% D(t,i) is a vector of weights, with one weight for eac training example i
% D(t+1,i) = D(t,i) * exp(-alpha(t,:)*y(t,:)*output(t,:)) / sum(sum(D))
result = 0;
for i = 0:numClassifiers
	result = result + alpha(i,:) * output(i,:)
end

sign(result)

% T = # base classifiers
% d_1(i) = 1/M
% errors - vector of errors for each base classifier
% for t = 1:T
%	error = 0;
%	for i = 1:M	
%		error = error + d(t,i) * I(y(i,:) != output(t,X(i,:)))
%	end
%	errors(t,:) = error
%	d(t+1,i) = d(t,i)*exp(-alpha(t,:)*y(i,:)* output(t,X(i,:)))
%   d(t+1,i) = sum(d(t+1,:))
%end
%Output(x) = sign(sum(alpha*output(t,x)))
%