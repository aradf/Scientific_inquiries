
%%% This document is an engineering check list to provide code development for the Object Detection.
%%% [REFERENCE 00-1] [REFERENCE 00-2] [REFERENCE 00-3] [REFERENCE 00-4] [REFERENCE 00-5]
%%% [REFERENCE 00-6] [REFERENCE 00-7] [REFERENCE 00-8]

%%%  Step 1: Data Pre-processing Image Resizing
%%% Resize image to 416x416x3 to account for the YOLOv2 Architecture. 
image_resizing_needed = false;
if image_resizing_needed
    
    %%% Enter the name of the folder that holds your dataset
    folder_name = '/home/faramarz/matlab_R2020b_glnxa64/NumberImages';
    
    src_dir = fullfile(folder_name)
    src_files = dir([src_dir,'*.jpeg']);
    for i = 1 : length(src_files)
        file_name = [src_dir ,src_files(i).name];
        im = imread(file_name);
        k=imresize(im,[416,416]);
        new_file_name=[src_dir ,src_files(i).name];
        imwrite(k,new_file_name,'jpeg');
    end
end

%%%  Step 2: Split Folders 
%%% Split the validation dataset folder into 2, one for testing and one for validation. The dataset 
%%% provided is already split randomly using the code below, run this code if you are using your own dataset.
split_folder = false;
if split_folder
    val_folder_name = Enter text;
    
    files = dir(fullfile(val_folder_name));
    N = length(files);
    tf=randperm(N)>(0.50*N);
    mkdir test_resized;
    mkdir val_resized;
    
    for i=3:length(tf)
        files_re = files(i).name;
        if tf(i) == 1
            copyfile (fullfile(val_folder_name,files_re),'test_resized')
        else
            copyfile (fullfile(val_folder_name,files_re),'val_resized')
        end
    end
end


%%% Step 3: Create Ground Truth
%%% Use the Ground truth labeler app to label the objects of interest in the dataset. Watch this 
%%% 5-minute video to learn how to use the Ground Truth Labeler App

is_ground_truth_available = false;
if is_ground_truth_available
    labeling_session_name = fullfile("NumberImages","number_image_label_session.mat");
    %%% groundTruthLabeler(labeling_session_name)
    imageLabeler(labeling_session_name)
else
    %%% groundTruthLabeler;
    imageLabeler;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Aggregate channel features for multi-view number detection (ACF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Create Training Data from Ground Truth
%%% Isolate ground truth data by their association with the 'one' label. This creates a new gTruth object 
%%% that contains the ground truth of only the one

one_label = selectLabels(gTruth,'one')

%%% Create a folder named TrainingData in the current folder to store the training images and add to the path

if isfolder(fullfile('TrainingDataACF'))
    cd TrainingDataACF
else
    mkdir TrainingDataACF
end 

addpath('TrainingDataACF');

%%% Extract a subset of the Ground Truth dataset. trainingData is a table that contains training data from the 
%%% ground truth. This table can be used with the inbuilt training functions like trainACFObjectDetector, 
%%% trainRCNNObjectDetector, trainFastRCNNObjectDetector etc.

training_data_acf = objectDetectorTrainingData(one_label,'SamplingFactor',2, 'WriteLocation','TrainingDataACF');

%%% Train the Detector
%%% Train the ACF detector. You can turn off the training progress output by specifying 'Verbose',false as 
%%% a Name,Value pair. 

%%% detector_acf = trainACFObjectDetector(training_data_acf,'NegativeSamplesFactor',2);
%%% detector_acf = trainACFObjectDetector(training_data_acf,'NumStages',25);
detector_acf = trainACFObjectDetector(training_data_acf,'NumStages',5);
cd TrainingDataACF/
save('Detector_acf.mat','detector_acf');
cd ..
rmpath('TrainingDataACF');

img = imread('NumberImages/images406.jpeg');
%%% [bboxes,scores] = detect(detector_acf, img, 'Threshold',1);
[bboxes,scores] = detect(detector_acf, img);

for i = 1:length(scores)
   if (scores(i)>1.0)
     annotation = sprintf('Confidence = %.1f',scores(i));
     img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
   end
end

figure
imshow(img)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The single shot multibox detector (SSD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

one_label = selectLabels(gTruth,'one')
%%% imds = imageDatastore(trainingData.imageFilename);
image_data_store_digit = imageDatastore(one_label.DataSource.Source)


table_size = size (gTruth_table);

new_cell = cell(table_size(1),1)
for iCnt = 1:table_size(1)
	a1 = gTruth_table.one(iCnt)
	a2 = gTruth_table.two(iCnt)
	a3 = gTruth_table.three(iCnt)
	a4 = gTruth_table.four(iCnt)
	a5 = gTruth_table.five(iCnt)
	a6 = gTruth_table.six(iCnt)
	a7 = gTruth_table.seven(iCnt)
	a8 = gTruth_table.eight(iCnt)
	a9 = gTruth_table.nine(iCnt)
	a0 = gTruth_table.zero(iCnt)
	new_cell{iCnt} = [a1{1};a2{1};a3{1};a4{1};a5{1};a6{1};a7{1};a8{1};a9{1};a0{1}] 
end

%%% Create an image datastore using the files from the table.
%%% imds = imageDatastore(trainingData.imageFilename);
imds_number = imageDatastore(gTruth_table(:,1).imageFilename)

%%% Create a box label datastore using the label columns from the table.
%%% blds = boxLabelDatastore(trainingData(:,2:end));
blds_number = boxLabelDatastore(cell2table(gTruth_table.number));
blds_number = boxLabelDatastore(gTruth_table(:,2:end))

%%% Combine the datastores.
ds_number = combine(imds_number, blds_number);


%%% Load a preinitialized SSD object detection network.
inputSize = [300 300 3];
numClasses = 1
%%% Create the SSD object detection network.

lgraph = ssdLayers(inputSize, numClasses, 'resnet50');
lgraph.Layers

%%% Configure the network training options.
options = trainingOptions('sgdm',...
          'InitialLearnRate',5e-5,...
          'MiniBatchSize',1,...
          'Verbose',true,...
          'MaxEpochs',50,...
          'Shuffle','every-epoch',...
          'VerboseFrequency',10,...
          'CheckpointPath',tempdir);

%%% Train the SSD network.
[detector,info] = trainSSDObjectDetector(ds_number,lgraph,options);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Create an image datastore using the files from the table.
imds_number = imageDatastore(gTruth_table.imageFilename);

%%% Create a box label datastore using the label columns from the table.
blds_number = boxLabelDatastore(cell2table(gTruth_table.number));

%%% Combine the datastores.
ds_number = combine(imds_number, blds_number);

%%% Load a preinitialized SSD object detection network.
net = load('ssdVehicleDetector.mat');
lgraph = net.lgraph
lgraph.Layers

%%% Configure the network training options.
options = trainingOptions('sgdm',...
          'InitialLearnRate',5e-5,...
          'MiniBatchSize',1,...
          'Verbose',true,...
          'MaxEpochs',50,...
          'Shuffle','every-epoch',...
          'VerboseFrequency',10,...
          'CheckpointPath',tempdir);

%%% Train the SSD network.

[detector,info] = trainSSDObjectDetector(ds_number,lgraph,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% g_Truth_Table.DataSource.Source
%%% g_Truth_Table.LabelData

imageFilename = g_Truth_Table.DataSource.Source
imageFilename = cell2table(imageFilename)
number = g_Truth_Table.LabelData.number
%%% imageFilename = g_Truth_Table.DataSource.Source{:,:}
%%% number = g_Truth_Table.LabelData.number{:,:}
number_dataset=table(imageFilename,number)
number_dataset(1:4,:)



imds_train = imageDatastore(number_dataset{:,'imageFilename'});
blds_train = boxLabelDatastore(number_dataset(:,'number'));
training_data = combine(imds_train, blds_train);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect objects using template matching, histogram of 
%%% gradients (HOG), and cascade object detectors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
target = imread('NumberImages/images415.jpeg');
imshow(target)

back_ground = imread('NumberImages/images367.jpeg');
imshow(back_ground)

back_ground_gray=rgb2gray(back_ground);
imshow(back_ground_gray)

target_gray=rgb2gray(target);
imshow(target_gray)

target_gray_resized = imresize(target_gray,[50,50]);
imshow(target_gray_resized)

H = vision.TemplateMatcher
H.SearchMethod = 'Three-step'

loc = step(H,back_ground_gray,target_gray_resized)
J = insertMarker(back_ground_gray,loc,'square','Size',10);
imshow(J)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Specify the folder for negative images.
%%% Specify false alarm rate

ground_truth = table2struct(ground_truth_table);
negativeFolder = '/home/faramarz/matlab_R2020b_glnxa64/bikeNoneImages'
num_stages = 5;
FAR = 0.01;

ground_truth_table.Properties.VariableNames{2} = 'objectBoundingBoxes';

trainCascadeObjectDetector('number_detector.xml',ground_truth_table, ...
    negativeFolder, 'FalseAlarmRate', FAR, 'NumCascadeStages', num_stages);

%%% Use the newly trained classifier to detect a stop sign in an image.
detector = vision.CascadeObjectDetector('number_detector.xml');


%%% Read the test image.
img_target = imread('NumberImages/images300.jpeg');

%%% Detect a stop sign.
bbox = step(detector, img_target);

%%% Insert bounding box rectangles and return the marked image.
%%% detected_img = insertObjectAnnotation(img_target,'rectangle',bbox,'Number');
detected_img = insertShape(img_target, 'rectangle', bbox);

%%% Display the detected stop sign.

figure; imshow(detected_img);
release(detector);


https://www.youtube.com/watch?v=J9HcjS9j2oY


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% You Only Look Once for Number Detection (YOLOV2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf



%%% Related sources:
%%% [REFERENCE 00-1] https://blogs.mathworks.com/racing-lounge/2019/04/11/designing-object-detectors-in-matlab/?s_eid=PSM_15028
%%% [REFERENCE 00-2] https://www.mathworks.com/matlabcentral/fileexchange/73954-deep-learning-for-object-detection?s_eid=PSM_15028
%%% [REFERENCE 00-3] https://www.mathworks.com/help/vision/ref/ssdobjectdetector.detect.html
%%% [REFERENCE 00-4] https://www.mathworks.com/help/vision/ug/object-detection-using-single-shot-detector.html
%%% [REFERENCE 00-5] https://www.mathworks.com/help/vision/ref/trainssdobjectdetector.html
%%% [REFERENCE 00-6] https://www.mathworks.com/help/vision/ug/create-ssd-object-detection-network.html
%%% [REFERENCE 00-7] https://www.youtube.com/watch?v=bRcX8l17ayU
%%% [REFERENCE 00-8] https://www.youtube.com/watch?v=Xe4tCLcSS1c
%%% [REFERENCE 00-9] https://www.mathworks.com/help/vision/ug/train-a-cascade-object-detector.html
%%% [REFERENCE 0-10] https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf




