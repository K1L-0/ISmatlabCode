clc;
% Load the training and testing data
TrainingData = imageDatastore('TrainingFolder', 'IncludeSubfolders', true, 'LabelSource', 'FolderNames');
TestingData = imageDatastore('TestingFolder', 'IncludeSubfolders', true, 'LabelSource', 'FolderNames');

HogCell = [8 8];
HogFeatures = [];

for i = 1:size(TrainingData.Files, 1)
    img = readimage(TrainingData, i);
    img = imresize(img,[64 64]);
    Images = extractHOGFeatures(img, 'CellSize', HogCell);
    HogFeatures = [HogFeatures; Images];
end


TrainingEmotions = TrainingData.Labels;

tic
SVMTraining = fitcecoc(HogFeatures, TrainingEmotions, 'Coding', 'onevsone');
toc

TestImages = [];
for i = 1:size(TestingData.Files, 1)
    img = readimage(TestingData, i);
    img = imresize (img, [64 64]);
    Images = extractHOGFeatures(img, 'CellSize', HogCell);
    TestImages = [TestImages; Images];
end

% Get the Testing labels
TestingEmotions = TestingData.Labels;

% Test the SVM on the testing data
tic
SVMPredictedEmotions = predict(SVMTraining, TestImages);
toc

% Calculate the accuracy of the Trained SVM
svmAccuracy = sum(SVMPredictedEmotions == TestingEmotions) / numel(TestingEmotions);

ResultsMatrix = confusionmat(TestingEmotions, SVMPredictedEmotions);
confusionchart(ResultsMatrix, unique(TestingEmotions));