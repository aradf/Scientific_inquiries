

%%%% Load the training data for vehicle detection into the workspace.
data = load('vehicleTrainingData.mat');
training_data = data.vehicleTrainingData;

%%%% Specify the directory in which training samples are stored
data_dir = fullfile(toolboxdir('vision'),'visiondata');
training_data.imageFilename = fullfile(data_dir,training_data.imageFilename);

%%%% Randomly shuffle data for training.
rng(0);
shuffled_idx = randperm(height(training_data));
training_data = training_data(shuffled_idx,:);


%%%% Create an imageDatastore using the files from the table.
imds = imageDatastore(training_data.imageFilename);


%%%% Create a boxLabelDatastore using the label columns from the table.
blds = boxLabelDatastore(training_data(:,2:end));


%%%% Combine the datastores.
ds = combine(imds, blds);

%%%% Load a preinitialized YOLO v2 object detection network.
net = load('yolov2VehicleDetector.mat');
lgraph = net.lgraph

%%%% Inspect the layers in the YOLO v2 network and their properties. Create the 
%%%% YOLO v2 network by following the steps given in Create YOLO v2 Detection Network.

lgraph.Layers

%%%% Configure the network training options.

options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',16,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);

%%%% Train the YOLO v2 network.

[detector,info] = trainYOLOv2ObjectDetector(ds,lgraph,options);

detector
info

%%%% Verify the training accuracy by inspecting the training loss for each iteration.
figure
plot(info.TrainingLoss)
grid on
xlabel('Number of Iterations')
ylabel('Training Loss for Each Iteration')


%%%% Read a test image into the workspace.
img = imread('detectcars.png');
%%%% img = imread('car_01.png');

%%%% Run the trained YOLO v2 object detector on the test image for vehicle detection.
[bboxes,scores] = detect(detector,img);
%%%% Display the detection results.

if(~isempty(bboxes))
    img = insertObjectAnnotation(img,'rectangle',bboxes,scores);
end
figure
imshow(img)


