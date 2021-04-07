
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% https://www.mathworks.com/matlabcentral/fileexchange/69180-using-ground-truth-for-object-detection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Using Ground Truth for Object Detection: Part (1)
viReader = VideoReader('BuoyRun2GT.mp4');
viPlayer = vision.DeployeableVideoPlayer;
while(hasFrame(vidReader))
     I = readFrame(vidReader);
     step(vidPlayer,I);
     pause(0.05);
end

%%% groundTruthLabeler;
cd UsingGroundTruthForObjectDetectionFX/
cd Part2/
videoLabeler;
videoLabeler('BuoyRun.avi');
imageLabeler;
redBuoyGTruth = selectLabels(gTruth,'bigRedBuoy');

%%% Open a Pre-Labeled Ground Truth Session
groundTruthLabeler('groundTruthLabelingSessionPart1.mat');

%%% Save labels to a MAT file
save('gTruthTraining.mat','gTruth');
load('gTruthTraining.mat');

%%% Create Training Data from Ground Truth.  Isolate ground truth 
%%% data by their association with the 'bigRedBuoy' label. This creates 
%%% a new gTruth object that contains the ground truth of only the bigRedBuoy
redBuoyGTruth = selectLabels(gTruth,'bigRedBuoy');

%%% Create a folder named TrainingData in the current folder to store 
%%% the training images and add to the path
if isfolder(fullfile('TrainingData'))
    cd TrainingData
else
    mkdir TrainingData
end 
addpath('TrainingData');

%%% Extract a subset of the Ground Truth dataset. trainingData is a table 
%%% that contains training data from the ground truth. This table can be 
%%% used with the inbuilt training functions like trainACFObjectDetector, 
%%% trainRCNNObjectDetector, trainFastRCNNObjectDetector etc.
trainingData = objectDetectorTrainingData(redBuoyGTruth,'SamplingFactor',2,'WriteLocation','TrainingData');

%%% Train the Detector
%%% Train the ACF detector. You can turn off the training progress 
%%% output by specifying 'Verbose',false as a Name,Value pair.
detector = trainACFObjectDetector(trainingData,'NumStages',5);

%%% Save the detector to a MAT file
save('Detector1.mat','detector');
rmpath('TrainingData');

close all
clear
clc

%%% Load a pre-trained ACF detector.
load('Detector.mat');

%%% Create video file reader for input
vidReader = VideoReader('BuoyRun.avi');
%%% Create video player for visualization
vidPlayer = vision.DeployableVideoPlayer;
%%% Initialise variables 
i = 1;
results = struct('Boxes',[],'Scores',[])

while(hasFrame(vidReader))    
    %%% GET DATA
    I = readFrame(vidReader);    
    
    %%% PROCESS
    [bboxes, scores] = detect(detector,I,'Threshold',1);
    
    %%% Select strongest detection 
    [~,idx] = max(scores);
    results(i).Boxes = bboxes;
    results(i).Scores = scores;
    
    %%% VISUALIZE
    annotation = sprintf('%s , Confidence %4.2f',detector.ModelName,scores(idx));
    I = insertObjectAnnotation(I,'rectangle',bboxes(idx,:),annotation);
    step(vidPlayer,I);
    i = i+1;   
end
results = struct2table(results);
release(vidPlayer);

%%% Load Ground Truth Data
%%% Load the gTruth object from a saved MAT file. This dataset is different than 
%%% the dataset used to train the detector. The ground truth is the set of known 
%%% locations of objects of interest in a set of images to be used to validate the detectors
load('gTruthEvaluation.mat')

%%% Create Dataset from Ground Truth
%%% Isolate ground truth data by their association with the 'bigRedBuoy' label. This creates 
%%% a new gTruth object that contains the ground truth of only the bigRedBuoy

redBuoyGTruth = selectLabels(gTruth,'bigRedBuoy');

%%% Create a folder named EvaluationData in the current folder, if EvaluationData dosent exist, to 
%%% store the training images and add to the path. Extract a subset of the Ground Truth dataset. 
%%% evaluationData is a table that contains training data from the ground truth. This table 
%%% can be used with the inbuilt evaluation functions like 
if ~isfolder(fullfile('EvaluationData'))
    
    mkdir EvaluationData
    addpath('EvaluationData');
    
    evaluationData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,'WriteLocation','EvaluationData');
end

%%% Construct an Image Datastore 
imds = imageDatastore(fullfile('EvaluationData'));

%%% Construct structure to store detection results
numImages = height(evaluationData);
result(numImages,:) = struct('Boxes',[],'Scores',[]);

%%% Process dataset to obtain detection results
for i = 1:numImages
    
    %%% Read Image
    I = readimage(imds,i); 
    
    %%% Detect the object of interest
    [bboxes, scores] = detect(detector,I,'Threshold',1);
    
    %%% Store result 
    result(i).Boxes = bboxes;
    result(i).Scores = scores;
   
end
%%% Convert structure to table
results = struct2table(result);

%%% Evaluate Detector
overlap = 0.5;

%%% Evaluate Metrics
[ap,recall,precision] = evaluateDetectionPrecision(results, evaluationData(:,3), overlap);
[am,fppi,missRate] = evaluateDetectionMissRate(results,evaluationData(:,3),overlap);

%%% Plot Metrics
subplot(1,2,1);
plot(recall,precision);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Average Precision = %.1f', ap))
grid on
subplot(1,2,2);
loglog(fppi, missRate);
xlabel('False Positives Per Image');
ylabel('Log Average Miss Rate');
title(sprintf('Log Average Miss Rate = %.1f', am))
grid on

%%%Clean Up
rmpath('EvaluationData');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% https://www.mathworks.com/matlabcentral/fileexchange/73954-deep-learning-for-object-detection?s_eid=PSM_15028
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1: Down load data.
%%% Step 2: Image Resizing
%%% Resize image to 416x416x3 to account for the YOLOv2 Architecture. For information on this click 
%%% this link. The dataset provided has already been resized to make it easier for sharing, this 
%%% step needs to be done only if you are using your own data. 
clear;
imageResizingNeeded = true;
if imageResizingNeeded
    folderName = '/home/faramarz/matlab_R2020b_glnxa64/HouseNumberImages/train'; % Enter the name of the folder that holds your dataset
    
    srcDir = fullfile(folderName)
    srcFiles = dir([srcDir,'*.png']);
    for i = 1 : length(srcFiles)
        filename = [srcDir ,srcFiles(i).name];
        im = imread(filename);
        k=imresize(im,[416,416]);
        newfilename=[srcDir ,srcFiles(i).name];
        imwrite(k,newfilename,'jpeg');
    end
end

%%% Step 3: Split Folders
%%% Split the validation dataset folder into 2, one for testing and one for validation. The dataset 
%%% provided is already split randomly using the code below, run this code if you are using your 
%%% own dataset. 
splitFolder = false;
if splitFolder
    valFolderName = 'Enter text';
    
    files = dir(fullfile(valFolderName));
    N = length(files);
    tf=randperm(N)>(0.50*N);
    mkdir testResized;
    mkdir valResized;
    
    for i=3:length(tf)
        files_re = files(i).name;
        if tf(i) == 1
            copyfile (fullfile(valFolderName,files_re),'testResized')
        else
            copyfile (fullfile(valFolderName,files_re),'valResized')
        end
    end
end

%%% Step 4: Create Ground Truth
%%% Use the Ground truth labeler app to label the objects of interest in the dataset. 
isGroundTruthAvailable = false;
if isGroundTruthAvailable
    labelingSessionName = fullfile("Utilities","groundTruthLabelingSessionRoboSubResizedTrain.mat");
    groundTruthLabeler(labelingSessionName)
else
    groundTruthLabeler;
end

%%% Step 5: Convert gTruth object into Training, Validation and Testing Data
%%% Use the splitLabels function to isolate the ground truth data for each class

if ~exist('gTruthResizedTrain', 'var')
    %%% If a data source warning is thrown in the command window, run adjustGroundTruthPaths.m
    load (fullfile("HouseNumberImages","HouseNumberTrainResized.mat")) 
end

%%% Create Training Data from Ground Truth Data
if ~isfolder(fullfile("TrainingData"))
    mkdir TrainingData
end

trainingData = objectDetectorTrainingData(gTruthResizedTrain, 'SamplingFactor', 1, 'WriteLocation', 'TrainingData');
%%% testData = objectDetectorTrainingData(gTruthResizedTest, 'SamplingFactor', 1, 'WriteLocation', 'TestData');

%%% Display first few rows of the data set.
trainingData(1:4,:)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Run this script to adjust the paths for all the gTruth Data Objects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

load('gTruthResizedVald.mat');
%%% original datasource path, do not edit
oldPathDataSource = fullfile("C:","Sandbox","DLODFX","deepLearningForObjectDetection","RoboSubFootage","valResized");

%%% new datasource path, edit with the appropriate location
newPathDataSource = fullfile("RoboSubFootage","valResized");
alterPaths = {[oldPathDataSource newPathDataSource]};
unresolvedPaths = changeFilePaths(gTruthResizedVal,alterPaths);
cd Utilities
save('gTruthResizedVald.mat', 'gTruthResizedVal');
clear gTruthResizedVal
cd ..

%%%
load('gTruthResizedTest.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% https://www.mathworks.com/matlabcentral/fileexchange/73954-deep-learning-for-object-detection?s_eid=PSM_15028
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Step 1: Design YOLOv2 network layers
%%% In this model we designed a custom YOLOv2 model layer by layer. The model starts by an input layer. 
%%% Then the detection subnetwork contains a series of Conv, Batch norm, ReLu layers, and maxPooling Layer
%%% followed by the, yolov2TransformLayer and yolov2OutputLayer objects, respectively.  The yolov2TransformLayer
%%% transforms the raw CNN output into a form required to produce object detections.  The yolov2OutputLayer 
%%% defines the anchor box parameters and implements the loss function used to train the detector.

%%% The imageInputLayer function is used to define the image input layer with minimum image size (128x128x3 used
%%% here). Use your best judgement based on the dataset and objects that need to be detected.

inputLayer = imageInputLayer([128 128 3], 'Name', 'input', 'Normalization', 'none');

%%% Set the convolution layer filter size to [3 3]. This size is common in CNN architectures. FilterSize 
%%% defines the size (height & width) of the local regions to which the neurons connect in the input.

filterSize = [3 3];

%%% For the middle layers we followed the basic approach of YOLO9000 paper and used a a repeated batch of
%%% Convolution2dLayer, Batch Normalization Layer, RelU Layer and Max Pooling Layer. Similar to the other
%%% classification models like VGG model we doubled the no of channels (filterSize) after every pooling 
%%% step. We used batch normalization to stabilize training, speed up convergence and regularize the model. 

middleLayers = [
    convolution2dLayer(filterSize, 16, 'Padding', 1, 'Name', 'conv_1', 'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2,'Name', 'maxpool1')
    convolution2dLayer(filterSize, 32, 'Padding', 1, 'Name', 'conv_2', 'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(filterSize, 64, 'Padding', 1, 'Name', 'conv_3', 'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    convolution2dLayer(filterSize, 128, 'Padding', 1, 'Name', 'conv_4', 'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'relu_4')
    ];

%%% Step 2: Create layer graph for yolov2 network.
%%% Combine the initial & middle layers and convert into a layer graph object in order to manipulate the layers

lgraph = layerGraph([inputLayer; middleLayers]);

%%% Compute number of Classes based on Input data.
numClasses = size(trainingData,2)-1;

%%% Step 3: Define Anchor boxes
%%% Anchor boxes are a set of predefined bounding boxes of a certain height and width. These boxes are defined 
%%% to capture the scale and aspect ratio of specific object classes you want to detect and are typically chosen 
%%% based on object sizes in your training datasets. You can define several anchor boxes, each for a different 
%%% object size. The use of anchor boxes enables a network to detect multiple objects, objects of different 
%%% scales, and overlapping objects. You can study in details about the Basics of anchor boxes here. 
%%% The anchor boxes is selected based on the scale and size of objects in the training data. You can 
%%% Estimate Anchor Boxes Using Clustering to determine a good set of anchor boxes based on the training data. 
%%% Using this procedure, the anchor boxes for the dataset are:

open(fullfile("AnchorBoxes.m"));

%%% anchorBoxes — Anchor boxes M-by-2 matrix
%%% Anchor boxes, specified as an M-by-2 matrix defining the size and the number of anchor boxes. Each row 
%%% in the M-by-2 matrix denotes the size of the anchor box in the form of [height width]. M denotes the number 
%%% of anchor boxes. This input sets the AnchorBoxes property of the output layer.

%%% The size of each anchor box is determined based on the scale and aspect ratio of different object classes 
%%% present in input training data. Also, the size of each anchor box must be smaller than or equal to the 
%%% size of the input image. You can use the clustering approach for estimating anchor boxes from the 
%%% training data. For more information, see Estimate Anchor Boxes From Training Data.

%%% https://www.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html
trainingDataStore = boxLabelDatastore(trainingData(:,2:end));
[Anchors,meanIoU] = estimateAnchorBoxes(trainingDataStore,numAnchors);

%%% An example of Anchors could be as follow (note no value is larger than 129).
Anchors = [43 59 
           18 22 
           23 29 
           84 109];

Anchors = [128    54
           127    28
           128    38
           128    73];

%%% Step 4: Assemble YOLOv2 network
%%% The yolov2Layers function creates a YOLO v2 network, which represents the network architecture for YOLO v2 
%%% object detector.  'relu_4' is the feature extraction layer , The features extracted from this layer are 
%%% given as input to the YOLO v2 object detection subnetwork. You can specify any network layer except the fully 
%%% connected layer as the feature layer.

lgraph = yolov2Layers([128 128 3],numClasses,Anchors,lgraph,'relu_4');

%%% Visualize the lgraph using the network analyzer app.
analyzeNetwork(lgraph);

%%% Step 5: Train the Network
%%% Set a random seed to ensure example training reproducibility.
doTraining = true; 

%%% setting this flag to true will build and train a YOLOv2 detector false will load a pre-trained network
if doTraining
    rng(0);

%%% Based on the size of the data set we train the network with the solver - stochastic gradient descent 
%%% for 80 epochs with initial learning rate of 0.001 and mini-batch size of 16. We performed lower learning 
%%% rate to give more time for training considering the size of data and adjusting the epoch and mini-batch 
%%% size. The other approcah could be more epochs like 160 with a higher learning rate of 0.01.  Detailed 
%%% Documentation to learn about all the options: trainingOptions.
%%% ExecutionEnvironment defines the Hardware resource for training network.
%%% DispatchInBackground is used only when parallel training or multi-gpu environment. 
options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.001, ...
        'Verbose',true,'MiniBatchSize',16,'MaxEpochs',80,...
        'Shuffle','every-epoch','VerboseFrequency',50, ...
        'DispatchInBackground',true,...
        'ExecutionEnvironment','auto');

%%% Call the YOLOv2 training function - trainYOLOv2ObjectDetector
[detectorYolo2, info] = trainYOLOv2ObjectDetector(trainingData,lgraph,options);    
else
    load(fullfile("Utilities","detectorYoloV2.mat")); %pre-trained detector loaded from a MAT file
end

%%% Step 6: Detect ROI's with the detector
%%% Create a table to hold the reults
results = table('Size',[height(TestData) 3], ...
   'VariableTypes',{'cell','cell','cell'}, ...
   'VariableNames',{'Boxes','Scores', 'Labels'});

%%% Initialize a Deployable Videl Player to view the image stream
depVideoPlayer = vision.DeployableVideoPlayer;

%%% Loop through all the images in the  Validation set
for i = 1:height(TestData)
    
    %%% Read the image
    I = imread(TestData.imageFilename{i});
    
    %%% Run the detector.
    [bboxes,scores,labels] = detect(detectorYolo2,I);
    
    %%%
    if ~isempty(bboxes)
        I = insertObjectAnnotation(I,'Rectangle',bboxes,cellstr(labels));
        depVideoPlayer(I);
        pause(0.3);
    end    
    
    %%% Collect the results in the results table
    results.Boxes{i} = floor(bboxes);
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    
end


%%% Step 6: Compute Evaluation metrics and plot the results
threshold = 0.5;

%%% To evaluate the precision metrics we use evaluateDetectionPrecision function. This function 
%%% here returns the data points for plotting the precison-recall curve using the given input 
%%% arguments and the threshold value. 
[ap, recall, precision] = evaluateDetectionPrecision(results, TestData(:,2:end),threshold);

%%% To evaluate miss rate metric for object detection we use the function evaluateDetectionMissRate.
%%% This function returns the log average -miss rate of the results compared to the groundTruthData 
%%% and data points fpr plotting the log miss rate to false positives per image.  

[am,fppi,missRate] = evaluateDetectionMissRate(results, TestData(:,2:end),threshold);

%%% Plot the evaluation metrics for each class
subplot(1,2,1);
plot(recall{1,1},precision{1,1},'g-','LineWidth',2, "DisplayName",'greenBuoy');
hold on;
plot(recall{2,1},precision{2,1},'b-','LineWidth',2, "DisplayName",'navGate');
hold on;
plot(recall{3,1},precision{3,1},'r-','LineWidth',2, "DisplayName",'redBuoy');
hold on;
plot(recall{4,1},precision{4,1},'y-','LineWidth',2, "DisplayName",'yellowBuoy');
hold off;
xlabel('Recall');
ylabel('Precision');
title(sprintf('Average Precision = %.2f\n', ap))
legend('Location', 'best');
legend('boxoff')
grid on

subplot(1,2,2);
loglog(fppi{1,1}, missRate{1,1},'-g','LineWidth',2, "DisplayName",'greenBuoy');
hold on;
loglog(fppi{2,1}, missRate{2,1},'-b','LineWidth',2,"DisplayName",'navGate');
hold on;
loglog(fppi{3,1}, missRate{3,1},'-r','LineWidth',2, "DisplayName",'redBuoy');
hold on;
loglog(fppi{4,1}, missRate{4,1},'-y','LineWidth',2, "DisplayName",'yellowBuoy');
hold off;
xlabel('False Positives Per Image');
ylabel('Log Average Miss Rate');
title(sprintf('Log Average Miss Rate = %.2f\n', am))
legend('Location', 'best');
legend('boxoff')
grid on

%%% https://www.mathworks.com/matlabcentral/fileexchange/73954-deep-learning-for-object-detection?s_eid=PSM_15028
%%% Import Pre-Trained Network from TensorFlow

%%% Introduction:  Tiny YOLOv2  is a real-time neural network for object detection that detects 20 different classes. 
%%% It has 9 convolutional layers and 6 max-pooling layers and is a smaller version of YOLOv2 network.

%%% Step 1: Download Tiny YOLOv2 ONNX model
%%% For this example download the TensorFlow Tiny YOLOv2 ONNX model from the prebuilt ONNX Models. Save 
%%% the downloaded model file as model.onnx (https://github.com/onnx/models) in the Utilities Folder.

%%% Step 2: Import network using ONNX
%%% Load the ONNX model file

modelfile = fullfile('tiny_yolo_v2','model.onnx');

%%% importONNXNetwork - Import a pretrained network from ONNX™ (Open Neural Network Exchange).
%%% OutputLayerType - Type of the output layer that the function appends to the end of the 
%%% imported network, specified as 'classification', 'regression', or 'pixelclassification'

net = importONNXNetwork(modelfile,'OutputLayerType','regression')
net = 
  DAGNetwork with properties:

         Layers: [34×1 nnet.cnn.layer.Layer]
    Connections: [33×2 table]
     InputNames: {'Input_image'}
    OutputNames: {'RegressionLayer_convolution8'}

%%% View the network configuration
analyzeNetwork(net)

%%% Step 3: Edit the Network using Deep Network Design App
%%% Open the deepNetworkDesigner app
useDesignApp = true;
if useDesignApp
    deepNetworkDesigner;
else
    load(fullfile('Utilities','importedTinyYoloLayers.mat'));
end 


%%% Utilizing the deepNetworkDesigner app:
%%% In the File section, click Import and choose the network net that we just imported from 
%%% ONNX above.  Edit the network by deleting the final regression layer  to prepare for transfer 
%%% learning Export the network to the MATLAB workspace - in the Export section, click Export.
%%% Deep Network Designer can generate MATLAB code for the layer definitions. In the Export 
%%% section, click Generate MATLAB Code to generate a live script with the actionable code.

%%% the exported network is saved as layers_1 in workspace
lgraph = layerGraph(layers_1);

%%% Compute number of Classes based on Input data.
numClasses = size(trainingData,2)-1;

%%% Define Anchor boxes
%%% Search for the key word 'AnchorBoxes.m' from above.
open(fullfile("AnchorBoxes.m"));
trainingDataStore = boxLabelDatastore(trainingData(:,2:end));
[Anchors,meanIoU] = estimateAnchorBoxes(trainingDataStore,numAnchors);

Anchors = [43 59
    18 22
    23 29
    84 109];

%%% Assemble YOLOv2 network
%%% The yolov2Layers function creates a YOLO v2 network, which represents the network architecture 
%%% for YOLO v2 object detector.  'convolution8' is the feature extraction layer , The features 
%%% extracted from this layer are given as input to the YOLO v2 object detection subnetwork. You 
%%% can specify any network layer except the fully connected layer as the feature layer.
%%%  adding the yolov2 layers with convolution8 as the feature layer

lgraph = yolov2Layers([128 128 3],numClasses,Anchors,net,'activation4');

%%% Visualize the lgraph using the network analyzer app.
analyzeNetwork(lgraph);

%%% Step 4: Perform Transfer Learning training
doTraining = true;

%%% setting this flag to true will perform transfer learning on the Tiny YOLOv2 detector
%%%  false will load a pre-trained network
if doTraining
    rng(0);

%%% Based on the size of the data set we train the network with the solver - stochastic gradient 
%%% descent for 80 epochs with initial learning rate of 0.001 and mini-batch size of 16. We performed 
%%% lower learning rate to give more time for training considering the size of data and adjusting 
%%% the epoch and mini-batch size. The other approcah could be more epochs like 160 with a higher 
%%% learning rate of 0.01.  Detailed Documentation to learn about all the options: trainingOptions.
%%% ExecutionEnvironment defines the Hardware resource for training network.  DispatchInBackground is 
%%% used only when parallel training or multi-gpu environment. 
       options = trainingOptions('sgdm', ...
        "LearnRateSchedule","piecewise",...
        'LearnRateDropFactor',0.5,...
        "LearnRateDropPeriod",5,...
        'Verbose',true,'MiniBatchSize',10,'MaxEpochs',100,...
        'Shuffle','every-epoch','VerboseFrequency',50, ...
        'DispatchInBackground',true,...
        'ExecutionEnvironment','auto');

%%% Call the YOLOv2 training function - trainYOLOv2ObjectDetector
    [detectorTinyYolo2, info] = trainYOLOv2ObjectDetector(table1,lgraph,options);
else
    load(fullfile('Utilities','detectorTinyYolo2.mat')); %pre-trained detector loaded from a MAT-file
end



