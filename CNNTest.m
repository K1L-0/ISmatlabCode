% Load the training and testing data
TrainingData = imageDatastore('TrainingFolder', 'IncludeSubfolders', true, 'LabelSource', 'FolderNames');
TestingData = imageDatastore('TestingFolder', 'IncludeSubfolders', true, 'LabelSource', 'FolderNames');

inputSize = [48 48 1];
numClasses = 7;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer([5 5], 64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5],64, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    convolution2dLayer([5 5], 16, Stride=2, Padding="same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...    
    'Verbose',false, ...
    'Plots','training-progress', ...
    InitialLearnRate=0.005,...
    MiniBatchSize=64, ...
    MaxEpochs=5);

tic
net = trainNetwork(TrainingData,layers,options);
toc

tic
PredictedEmotions = classify(net, TestingData); 
toc

ActualEmotions = TestingData.Labels;

accuracy = sum(PredictedEmotions == ActualEmotions) / numel(ActualEmotions);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
