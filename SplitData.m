%path to CK folder
PathToCK = "C:\Users\jacka\OneDrive\Documents\MATLAB\Intelligent Systems\Assessment\Dataset\CK+48";
EmotionFolders = dir(PathToCK);


EmotionFolders = EmotionFolders(~ismember({EmotionFolders.name},{'.','..'}));

%Variable for the Testing/Training split
TestSplit = 0.2;
imgSize = [64 64];

%Goes through each emotion stored in the CK+ directory and the images
%stored inside of them
for i = 1:length(EmotionFolders)
    EmotionFolder = fullfile(EmotionFolders(i).folder, EmotionFolders(i).name);
    ImageFolder = dir(fullfile(EmotionFolder, '*.png'));
    
    %Selects images to go into the Testing Folder
    TestNumber = round(TestSplit * length(ImageFolder));
    TestIndex = randperm(length(ImageFolder), TestNumber);
    
    %Creates the Testing and Training folder to store the split data
    TrainingDataFolder = fullfile("TrainingFolder", EmotionFolders(i).name);
    TestingDataFolder = fullfile('TestingFolder', EmotionFolders(i).name);

    %Checks the directory to see if the Training or Testing folder has
    %already been made, if it hasnt it will make the directory to add it
    if ~exist(TrainingDataFolder, 'dir')
        mkdir(TrainingDataFolder);
    end
    if ~exist(TestingDataFolder, 'dir')
        mkdir(TestingDataFolder);
    end
    
    % Loop that adds the images to the right folders 
    for j = 1:length(ImageFolder)

        %checks if the current image is part of the Testing index if so
        %will be added to the testing folder if not it will be added to the
        %training folder 
        if ismember(j, TestIndex)
            % resizes the image to the specified height and width
            ImageResize = imread(fullfile(EmotionFolder, ImageFolder(j).name));
            ImageResize = imresize(ImageResize, imgSize); 
            imwrite(ImageResize, fullfile(TestingDataFolder, ImageFolder(j).name));
            %copies the source file into the correct directory
            copyfile(fullfile(EmotionFolder, ImageFolder(j).name), TestingDataFolder);
        else
            ImageResize = imread(fullfile(EmotionFolder, ImageFolder(j).name));
            ImageResize = imresize(ImageResize, imgSize); 
            imwrite(ImageResize, fullfile(TestingDataFolder, ImageFolder(j).name));
            copyfile(fullfile(EmotionFolder, ImageFolder(j).name), TrainingDataFolder);
        end
    end
end

