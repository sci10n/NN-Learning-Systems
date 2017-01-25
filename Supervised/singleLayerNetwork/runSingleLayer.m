function [ Y, L ] = runSingleLayer(X, W)
%EVALUATESINGLELAYER Summary of this function goes here
%   Inputs:
%               X  - Features to be classified (matrix)
%               W  - Weights of the neurons (matrix)
%
%   Output:
%               Y = Output for each feature, (matrix)
%               L = The resulting label of each feature, (vector) 

Y = zeros(size(W,1),size(X,2));
L = zeros(1,size(X,2));
% Need to calculate each case by it self
   % for index = 1:size(X,2)
    % We also need to calculate for each neuron
    % Calculate the inputs with the weights and summarize
    Y = W * X;
    % Calculate the activation function for each of the weights
    %Y(:,index) = tanh(weighted_sum);
    % Calculate classified labels (Hint, use the max() function), get the
    % output neuron with the highest value
    [~,L] = max(Y);
    L = L(:);
 %   end
end

