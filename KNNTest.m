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

% Train the KNN model
tic
Neighbours = 3; % Number of nearest neighbors to consider
KNNTraining = fitcknn(HogFeatures, TrainingEmotions, 'NumNeighbors', Neighbours);
toc

TestImages = [];
for i = 1:size(TestingData.Files, 1)
    img = readimage(TestingData, i);
    img = imresize (img, [64 64]);
    Images = extractHOGFeatures(img, 'CellSize', HogCell);
    TestImages = [TestImages; Images];
end

% Get the testing labels
TestingEmotions = TestingData.Labels;

% Test the KNN model on the testing data
tic
PredictedEmotions = predict(KNNTraining, TestImages);
toc

% Calculate accuracy
accuracy = sum(PredictedEmotions == TestingEmotions) / numel(TestingEmotions);
fprintf('Accuracy = %.2f%%\n', accuracy * 100);

ResultsMatrix = confusionmat(TestingEmotions, PredictedEmotions);
confusionchart(ResultsMatrix, unique(TestingEmotions));