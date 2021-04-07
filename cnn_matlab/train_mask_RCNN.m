

%%% https://www.mathworks.com/help/vision/ug/object-detection-using-faster-r-cnn-deep-learning.html
%%% Download Pretrained Detector
%%% Download a pretrained detector to avoid having to wait for training to complete. If
%%% you want to train the detector, set the doTrainingAndEval variable to true.

doTrainingAndEval = false;
if ~doTrainingAndEval && ~exist('fasterRCNNResNet50EndToEndVehicleExample.mat','file')
    disp('Downloading pretrained detector (118 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/fasterRCNNResNet50EndToEndVehicleExample.mat';
    websave('fasterRCNNResNet50EndToEndVehicleExample.mat',pretrainedURL);
end

%%% Load Data Set
%%% This example uses a small labeled dataset that contains 295 images. Each image contains 
%%% one or two labeled instances of a vehicle. A small dataset is useful for exploring the 
%%% Faster R-CNN training procedure, but in practice, more labeled images are needed to train 
%%% a robust detector. Unzip the vehicle images and load the vehicle ground truth data.

unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

%%% The vehicle data is stored in a two-column table, where the first column contains the image 
%%% file paths and the second column contains the vehicle bounding boxes.
%%% Split the dataset into training, validation, and test sets. Select 60% of the data for 
%%% training, 10% for validation, and the rest for testing the trained detector.

rng(0)
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset));

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

%%% Use imageDatastore and boxLabelDatastore to create datastores for loading the image 
%%% and label data during training and evaluation.

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%%% Display one of the training images and box labels.

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%%% Create Faster R-CNN Detection Network
%%% A Faster R-CNN object detection network is composed of a feature extraction network followed 
%%% by two subnetworks. The feature extraction network is typically a pretrained CNN, such as 
%%% ResNet-50 or Inception v3. The first subnetwork following the feature extraction network 
%%% is a region proposal network (RPN) trained to generate object proposals - areas in the image 
%%% where objects are likely to exist. The second subnetwork is trained to predict the actual 
%%% class of each object proposal.

%%% The feature extraction network is typically a pretrained CNN. This example uses ResNet-50 
%%% for feature extraction. You can also use other pretrained networks such as MobileNet v2 or 
%%% ResNet-18, depending on your application requirements.

%%% Use fasterRCNNLayers to create a Faster R-CNN network automatically given a pretrained 
%%% feature extraction network. fasterRCNNLayers requires you to specify several inputs that 
%%% parameterize a Faster R-CNN network:

%%% First, specify the network input size. When choosing the network input size, consider the minimum 
%%% size required to run the network itself, the size of the training images, and the computational 
%%% cost incurred by processing data at the selected size. When feasible, choose a network input 
%%% size that is close to the size of the training image and larger than the input size required 
%%% for the network. To reduce the computational cost of running the example, specify a network 
%%% input size of [224 224 3], which is the minimum size required to run the network.

inputSize = [224 224 3];

%%% Note that the training images used in this example are bigger than 224-by-224 and vary 
%%% in size, so you must resize the images in a preprocessing step prior to training.

%%% Next, use estimateAnchorBoxes to estimate anchor boxes based on the size of objects in 
%%% the training data. To account for the resizing of the images prior to training, resize 
%%% the training data for estimating anchor boxes. Use transform to preprocess the training 
%%% data, then define the number of anchor boxes and estimate the anchor boxes.

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)


%%% Now, use resnet50 to load a pretrained ResNet-50 model.
featureExtractionNetwork = resnet50;

%%% Select 'activation_40_relu' as the feature extraction layer. This feature extraction layer outputs 
%%% feature maps that are downsampled by a factor of 16. This amount of downsampling is a good trade-off 
%%% between spatial resolution and the strength of the extracted features, as features extracted further 
%%% down the network encode stronger image features at the cost of spatial resolution. Choosing the optimal 
%%% feature extraction layer requires empirical analysis. You can use analyzeNetwork to find the names 
%%% of other potential feature extraction layers within a network.

featureLayer = 'activation_40_relu';

%%% Define the number of classes to detect.

numClasses = width(vehicleDataset)-1;

%%% Create the Faster R-CNN object detection network.

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%%% You can visualize the network using analyzeNetwork or Deep Network Designer from Deep Learning Toolbox™.

%%% Data Augmentation
%%% Data augmentation is used to improve network accuracy by randomly transforming the original 
%%% data during training. By using data augmentation, you can add more variety to the training 
%%% data without actually having to increase the number of labeled training samples.

%%% Use transform to augment the training data by randomly flipping the image and associated 
%%% box labels horizontally. Note that data augmentation is not applied to test and validation 
%%% data. Ideally, test and validation data are representative of the original data and are 
%%% left unmodified for unbiased evaluation.

augmentedTrainingData = transform(trainingData,@augmentData);


%%% Preprocess Training Data
%%% Preprocess the augmented training data, and the validation data to prepare for training.

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

%%%Read the preprocessed data.

data = read(trainingData);
%%% Display the image and box bounding boxes.

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%%% Train Faster R-CNN
%%% Use training Options to specify network training options. Set 'ValidationData' to the preprocessed 
%%% validation data. Set 'CheckpointPath' to a temporary location. This enables the saving of partially 
%%% trained detectors during the training process. If training is interrupted, such as by a power 
%%% outage or system failure, you can resume training from the saved checkpoint.

options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);

%%% Use trainFasterRCNNObjectDetector to train Faster R-CNN object detector if doTrainingAndEval 
%%% is true. Otherwise, load the pretrained network.

if doTrainingAndEval
    %% Train the Faster R-CNN detector.
    %% * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %%   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
else
    % Load pretrained detector for the example.
    pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
    detector = pretrained.detector;
end


%%% This example was verified on an Nvidia(TM) Titan X GPU with 12 GB of memory. Training the 
%%% network took approximately 20 minutes. The training time varies depending on the hardware you use.

%%% As a quick check, run the detector on one test image. Make sure you resize the image to the same
%%%  size as the training images.

I = imread(testDataTbl.imageFilename{4});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

%%% Display the results.

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

%%% https://www.mathworks.com/help/vision/ref/trainrcnnobjectdetector.html#d122e157246
%%% trainRCNNObjectDetector

%%% Load training data and network layers.
load('rcnnStopSigns.mat', 'stopSigns', 'layers')

%%% Add the image directory to the MATLAB path.

imDir = fullfile(matlabroot, 'toolbox', 'vision', 'visiondata', 'stopSignImages');
addpath(imDir);

%%% Set network training options to use mini-batch size of 32 to reduce GPU memory usage. 
%%% Lower the InitialLearningRate to reduce the rate at which network parameters are changed.

options = trainingOptions('sgdm', 'MiniBatchSize', 32, 'InitialLearnRate', 1e-6, 'MaxEpochs', 10);

%%% Train the R-CNN detector. Training can take a few minutes to complete.

rcnn = trainRCNNObjectDetector(stopSigns, layers, options, 'NegativeOverlapRange', [0 0.3]);

%%% Test the R-CNN detector on a test image.

img = imread('stopSignTest.jpg');

[bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);

%%% Display strongest detection result.

[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure
imshow(detectedImg)

%%% Remove the image directory from the path.
rmpath(imDir);

%%% Resume training an R-CNN object detector using additional data. To illustrate this 
%%% procedure, half the ground truth data will be used to initially train the detector. 
%%% Then, training is resumed using all the data.

%%% Load training data and initialize training options.
load('rcnnStopSigns.mat', 'stopSigns', 'layers')

stopSigns.imageFilename = fullfile(toolboxdir('vision'),'visiondata', stopSigns.imageFilename);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', 10, ...
    'Verbose', false);

%%% Train the R-CNN detector with a portion of the ground truth.

rcnn = trainRCNNObjectDetector(stopSigns(1:10,:), layers, options, 'NegativeOverlapRange', [0 0.3]);

%%% Get the trained network layers from the detector. When you pass in an array of network layers
%%% to trainRCNNObjectDetector, they are used as-is to continue training.

network = rcnn.Network;
layers = network.Layers;

%%% Resume training using all the training data.

rcnnFinal = trainRCNNObjectDetector(stopSigns, layers, options);

%%% Create a network for multiclass R-CNN object detection
%%% Create an R-CNN object detector for two object classes: dogs and cats.
objectClasses = {'dogs','cats'};

%%% The network must be able to classify both dogs, cats, and a "background" class in 
%%% order to be trained using trainRCNNObjectDetector. In this example, a one is 
%%% added to include the background.

numClassesPlusBackground = numel(objectClasses) + 1;

%%% The final fully connected layer of a network defines the number of classes that 
%%% the network can classify. Set the final fully connected layer to have an output 
%%% size equal to the number of classes plus a background class.

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)        
    fullyConnectedLayer(numClassesPlusBackground);
    softmaxLayer()
    classificationLayer()];

%%% These network layers can now be used to train an R-CNN two-class object detector.
%%% Load the stop sign training data.
load('rcnnStopSigns.mat','stopSigns','layers')

%%% Add full path to image files.
stopSigns.imageFilename = fullfile(toolboxdir('vision'),'visiondata', stopSigns.imageFilename);

%%% Set the 'CheckpointPath' using the trainingOptions function.

checkpointLocation = tempdir;
options = trainingOptions('sgdm','Verbose',false, 'CheckpointPath',checkpointLocation);

%%% Train the R-CNN object detector with a few images.
%%% Load a saved network checkpoint.

wildcardFilePath = fullfile(checkpointLocation,'convnet_checkpoint__*.mat');
contents = dir(wildcardFilePath);

%%% Load one of the checkpoint networks.

%%% https://www.mathworks.com/help/vision/ref/selectstrongestbboxmulticlass.html
%%% selectStrongestBboxMulticlass
%%% Run Multiclass Nonmaximal Suppression on Bounding Boxes Using People Detector

detectorInria = peopleDetectorACF('inria-100x41');
detectorCaltech = peopleDetectorACF('caltech-50x21');

%%% Apply the detectors.
I = imread('visionteam1.jpg');
[bboxesInria,scoresInria] = detect(detectorInria,I,'SelectStrongest',false);
[bboxesCaltech,scoresCaltech] = detect(detectorCaltech,I,'SelectStrongest',false);

%%% Create categorical labels for each the result of each detector.

labelsInria = repelem("inria",numel(scoresInria),1);
labelsInria = categorical(labelsInria,{'inria','caltech'});
labelsCaltech = repelem("caltech",numel(scoresCaltech),1);
labelsCaltech = categorical(labelsCaltech,{'inria','caltech'});


%%% Combine results from all detectors to for multiclass detection results.

allBBoxes = [bboxesInria;bboxesCaltech];
allScores = [scoresInria;scoresCaltech];
allLabels = [labelsInria;labelsCaltech];

%%% Run multiclass non-maximal suppression.

%%% Annotate detected people.
annotations = string(labels) + ": " + string(scores);
I = insertObjectAnnotation(I,'rectangle',bboxes,cellstr(annotations));
imshow(I)
title('Detected People, Scores, and Labels')

%%% https://www.mathworks.com/help/vision/ref/trainfasterrcnnobjectdetector.html
%%% trainFasterRCNNObjectDetector - Load training data.
data = load('fasterRCNNVehicleTrainingData.mat');
trainingData = data.vehicleTrainingData;
trainingData.imageFilename = fullfile(toolboxdir('vision'),'visiondata', trainingData.imageFilename);


%%% Randomly shuffle data for training.

rng(0);
shuffledIdx = randperm(height(trainingData));
trainingData = trainingData(shuffledIdx,:);

%%% Create an image datastore using the files from the table.
imds = imageDatastore(trainingData.imageFilename);

%%% Create a box label datastore using the label columns from the table.
blds = boxLabelDatastore(trainingData(:,2:end));

%%% Combine the datastores.
ds = combine(imds, blds);

%%% Set up the network layers.
lgraph = layerGraph(data.detector.Network)

%%% Configure training options.
options = trainingOptions('sgdm', 'MiniBatchSize', 1, 'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 7, 'VerboseFrequency', 200, 'CheckpointPath', tempdir);

%%% Train detector. Training will take a few minutes. Adjust the NegativeOverlapRange 
%%% and PositiveOverlapRange to ensure training samples tightly overlap with ground truth.

detector = trainFasterRCNNObjectDetector(trainingData, lgraph, options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);

%%% Test the Faster R-CNN detector on a test image.
img = imread('highway.png');

%%% Run the detector.
[bbox, score, label] = detect(detector,img);

%%% Display detection results.
detectedImg = insertShape(img,'Rectangle',bbox);
figure
imshow(detectedImg)


%%% Object Detection Using SSD Deep Learning
%%% https://www.mathworks.com/help/vision/ug/object-detection-using-single-shot-detector.html
%%% Download a pretrained detector to avoid having to wait for training to complete. If you 
%%% want to train the detector, set the doTraining variable to true.
doTraining = false;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end

%%% Load Dataset
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

vehicleDataset(1:4,:)

%%% Split the data set into a training set for training the detector and a test set for evaluating the detector
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );
trainingData = vehicleDataset(shuffledIndices(1:idx),:);
testData = vehicleDataset(shuffledIndices(idx+1:end),:);

%%% Use imageDatastore and boxLabelDatastore to load the image and label data during training and evaluation.
imdsTrain = imageDatastore(trainingData{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingData(:,'vehicle'));

imdsTest = imageDatastore(testData{:,'imageFilename'});
bldsTest = boxLabelDatastore(testData(:,'vehicle'));

%%% Combine image and box label datastores.

trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest, bldsTest);

%%% Display one of the training images and box labels.

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%%% Create a SSD Object Detection Network

%%% The SSD object detection network can be thought of as having two sub-networks. A feature 
%%% extraction network, followed by a detection network. 
inputSize = [300 300 3];

%%% Define number of object classes to detect.

numClasses = width(vehicleDataset)-1;

%%% Create the SSD object detection network.

lgraph = ssdLayers(inputSize, numClasses, 'resnet50');

%%% Data Augmentation
augmentedTrainingData = transform(trainingData,@augmentData);
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

%%% Read the preprocessed training data.

data = read(preprocessedTrainingData);

%%% Display the image and bounding boxes.

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%%% Train SSD Object Detector

%%% Use trainingOptions to specify network training options. Set 'CheckpointPath' to a temporary 
%%% location. This enables the saving of partially trained detectors during the training process. 
%%% If training is interrupted, such as by a power outage or system failure, you can resume 
%%% training from the saved checkpoint.

options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-1, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.8, ...
        'MaxEpochs', 300, ...
        'VerboseFrequency', 50, ...        
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch');


if doTraining
    % Train the SSD detector.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('ssdResNet50VehicleExample_20a.mat');
    detector = pretrained.detector;
end

%%% As a quick test, run the detector on one test image.

data = read(testData);
I = data{1,1};
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I, 'Threshold', 0.4);

%%% Display the results.

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)


%%% Create SSD Object Detection Network
%%% https://www.mathworks.com/help/vision/ug/create-ssd-object-detection-network.html

%%% TBD


%%% Getting Started with Mask R-CNN for Instance Segmentation
%%% https://www.mathworks.com/help/vision/ug/getting-started-with-mask-r-cnn-for-instance-segmentation.html
%%% https://github.com/matlab-deep-learning/mask-rcnn







