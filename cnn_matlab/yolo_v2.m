
%%%% Object dectection using Yolo V2 Deep Learning.
doTraining = false;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end

%%%% Load Data set
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicle_data_set = data.vehicleDataset;

%%%% Display first few rows of the data set.
vehicle_data_set(1:4,:)

%%%% Add the fullpath to the local vehicle data folder.
vehicle_data_set.imageFilename = fullfile(pwd,vehicle_data_set.imageFilename);

%%%% Split the dataset into training, validation, and test sets. Select 60% of 
%%%% the data for training, 10% for validation, and the rest for testing the trained detector.
rng(0);
shuffled_indices = randperm(height(vehicle_data_set));

idx = floor(0.6 * length(shuffled_indices) );

training_idx = 1:idx;
training_data_tbl = vehicle_data_set(shuffled_indices(training_idx),:);

validation_idx = idx+1 : idx + 1 + floor(0.1 * length(shuffled_indices) );
validation_data_tbl = vehicle_data_set(shuffled_indices(validation_idx),:);

test_idx = validation_idx(end)+1 : length(shuffled_indices);
test_data_tbl = vehicle_data_set(shuffled_indices(test_idx),:);

%%%% Use imageDatastore and boxLabelDatastore to create datastores for 
%%%% loading the image and label data during training and evaluation.
imds_train = imageDatastore(training_data_tbl{:,'imageFilename'});
blds_train = boxLabelDatastore(training_data_tbl(:,'vehicle'));

imds_validation = imageDatastore(validation_data_tbl{:,'imageFilename'});
blds_validation = boxLabelDatastore(validation_data_tbl(:,'vehicle'));

imds_test = imageDatastore(test_data_tbl{:,'imageFilename'});
blds_test = boxLabelDatastore(test_data_tbl(:,'vehicle'));

%%%% Combine image and box label datastores.
training_data = combine(imds_train,blds_train);
validation_data = combine(imds_validation,blds_validation);
test_data = combine(imds_test,blds_test);

%%%% Display one of the training images and box labels.
data = read(training_data);
I = data{1};
bbox = data{2};
annotated_image = insertShape(I,'Rectangle',bbox);
annotated_image = imresize(annotated_image,2);
figure
imshow(annotated_image)

%%%% Create a YOLO v2 Object Detection Network.  A YOLO v2 object detection 
%%%% network is composed of two subnetworks. A feature extraction network 
%%%% followed by a detection network. Define the number of object classes to detect.
input_size = [224 224 3];
num_classes = width(vehicle_data_set)-1;
training_data_for_estimation = transform(training_data,@(data)preprocessData(data,input_size));
num_anchors = 7;
[anchor_boxes, mean_iou] = estimateAnchorBoxes(training_data_for_estimation, num_anchors)

feature_extraction_network = resnet50;

%%%% Select 'activation_40_relu' as the feature extraction layer to replace the layers 
%%%% after 'activation_40_relu' with the detection subnetwork. 
feature_layer = 'activation_40_relu';
lgraph = yolov2Layers(input_size,num_classes,anchor_boxes,feature_extraction_network,feature_layer);

%%%% Data Augmentation.
augmented_training_data = transform(training_data,@augmentData);

%%%% Visualize the augmented images.
%%%% augmented_data = cell(4,1);
%%%% for k = 1:4
%%%%     data = read(augmented_training_data);
%%%%     augmented_data{k} = insertShape(data{1},'Rectangle',data{2});
%%%%     reset(augmented_training_data);
%%%% end
%%%% figure
%%%% montage(augmented_data,'BorderSize',10)


%%%% Preprocess Training Data
preprocessed_training_data = transform(augmented_training_data,@(data)preprocessData(data,input_size));
preprocessed_validation_data = transform(validation_data,@(data)preprocessData(data,input_size));

%%%% Read the preprocessed training data. Display the image and bounding boxes.
%%%% data = read(preprocessed_training_data);
%%%% I = data{1};
%%%% bbox = data{2};
%%%% annotated_image = insertShape(I,'Rectangle',bbox);
%%%% annotated_image = imresize(annotated_image,2);
%%%% figure
%%%% imshow(annotated_image)

%%%% Train YOLO v2 Object Detector
options = trainingOptions('sgdm', ...
                          'MiniBatchSize',16, ....
                          'InitialLearnRate',1e-3, ...
                          'MaxEpochs',20,...
                          'CheckpointPath',tempdir, ...
                          'ValidationData',preprocessed_validation_data);
doTraining = true;
if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessed_training_data,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end

%%% I = imread('highway.png');
I = imread('car_01.png');
I = imresize(I,input_size(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)


