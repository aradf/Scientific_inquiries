
%%% Code snippet courtesy of mathworks.  Yolo netowrk does not exist in the current 
%%% folder the first time you run the code, so download yolo network.   

clearvars -except yolonet yolo_ml yolojb
close all

if exist('yolonet.mat','file') == 0
    url = 'https://www.mathworks.com/supportfiles/gpucoder/cnn_models/Yolo/yolonet.mat';
    websave('yolonet.mat',url);
end

%%%load yolo network from current folder (this can take a while)
if exist('yolonet') ~= 1
    load yolonet.mat
end

%%% We will modify yolonet and save it with a new name (yolo_ml) before 
%%% the first time we run the script, 
if exist('yolo_ml.mat','file') == 0
    display('modifying yolo network')
    
    %%% extract a layer graph from the network. We need to modify this graph.
    lgraph = layerGraph(yolonet.Layers);
    
    %%% Modify the last two layers, since the yolo network from MATLAB is 
    %%% built like a classifier.  It must be regression network.
    lgraph = removeLayers(lgraph,'ClassificationLayer');
    lgraph = removeLayers(lgraph,'softmax');
    
    %%% According to the original YOLO paper, the last transfer function
    %%% is not a leaky, but a normal ReLu (I think).
    %%% In MATLAB, this is equivalent to a leaky ReLu with Scale = 0.
    alayer = leakyReluLayer('Name','linear_25','Scale',0);
    rlayer = regressionLayer('Name','routput');
    lgraph = addLayers(lgraph,rlayer);
    lgraph = replaceLayer(lgraph,'leakyrelu_25',alayer);
    lgraph = connectLayers(lgraph,'FullyConnectedLayer1','routput');
    yolo_ml = assembleNetwork(lgraph);
    
    %%%save the network with a new name
    display('saving modified network')
    save yolo_ml yolo_ml    
 
%%% if we have created and saved yolo_ml but not loaded it to workspace, load
%%% it now.
elseif exist('yolo_ml') ~= 1
    display('loading modified network')
    load('yolo_ml.mat')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prob_thresh = 0.10;
iou_thresh = 0.4;   

%%% preprocess an image: my input image must be resized to 448x448 pixels, and 
%%% convert from an unsigned 8bit to a single with pixel values scaled to [0,1].

dog_image = single(imresize(imread('Yolo_dog_01.jpeg'),[448 448]))/255;
figure(1);
imagesc(dog_image);

%%% Define 20 class labels that yolo has been trained on and in alphabetical order.
class_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", ...
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", ...
               "person",  "pottedplant", "sheep", "sofa", "train", "tvmonitor"];

%%% run the image through the network. Replace 'gpu' with 'cpu' if you do not
%%% have a CUDA enbled GPU.

tic
%%% dog_out = predict(yolo_ml, dog_image, 'ExecutionEnvironment', 'gpu');
dog_out = predict(yolo_ml, dog_image, 'ExecutionEnvironment', 'cpu');
toc

%%% plot the 1x1470 output vector. Indices 1-980 are class probabilities,
%%% 981-1079 are cell/box probabilities, and 1080-1470 are bounding box parameters
figure(2)
plot(dog_out)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% in order to make interpretation of this output vector more manageable
%%% with my finite visual-spatial skills, I decided to reshape the vector
%%% into a 7x7x30 array with the third dimension containing all information
%%% for each of the 7x7 cells.

class = dog_out(1:980);
box_prob = dog_out(981:1078);
box_dims = dog_out(1079:1470);

out_array = zeros(7,7,30);
temp = class;
    for j = 0:6
        for i = 0:6
            out_array(i+1,j+1,1:20) = class(i*20*7+j*20+1:i*20*7+j*20+20);
            out_array(i+1,j+1,21:22) = box_prob(i*2*7+j*2+1:i*2*7+j*2+2);
            out_array(i+1,j+1,23:30) = box_dims(i*8*7+j*8+1:i*8*7+j*8+8);
        end
    end

%%% find boxes with probabilities above a defined probability threshold. 0.2 seems to
%%% work well. cellIndex tells us which of two bounding boxes for each cell
%%% has higher probability. 1 for vertical box and 2 for horizontal box.

[cellProb cellIndex] = max(out_array(:,:,21:22),[],3);
contain = max(out_array(:,:,21:22),[],3)>prob_thresh;

%%% find highest probability object class type for each cell, save it to
%%% classMaxIndex
[classMax,classMaxIndex] = max(out_array(:,:,1:20),[],3);

figure(3)
imagesc(contain);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
%%%  put object containing box coordinates and other relevant information in a cell
%%% array
counter = 0;
for i = 1:7
    for j = 1:7
        if contain(i,j) == 1
            counter = counter+1;           
            
            %%% Bounding box center relative to cell
            x = out_array(i,j,22+1+(cellIndex(i,j)-1)*4);
            y = out_array(i,j,22+2+(cellIndex(i,j)-1)*4);
            
            %%% Yolo outputs the square root of the width and height of the
            %%% bounding boxes (subtle detail in paper that took me forver to realize). 
            %%% Relative to image size.
            w = (out_array(i,j,22+3+(cellIndex(i,j)-1)*4))^2;
            h = (out_array(i,j,22+4+(cellIndex(i,j)-1)*4))^2;
           
            %%% absolute values scaled to image size
            wS = w*448; 
            hS = h*448;
            xS = (j-1)*448/7+x*448/7-wS/2;
            yS = (i-1)*448/7+y*448/7-hS/2;
            
            %%% this array will be used for drawing bounding boxes in Matlab
            boxes(counter).coords = [xS yS wS hS]; 
            
            %%% save cell indices in the structure
            boxes(counter).cellIndex = [i,j];
            
            %%% save classIndex to structure
            boxes(counter).classIndex = classMaxIndex(i,j);    
            
            %%% save cell proability to structure
            boxes(counter).cellProb = cellProb(i,j);
            
            %%% put in a switch for non max which we will use later
            boxes(counter).nonMax = 1;
        end            
    end
end

%%% plot result without non-max suppression
figure(4) 
imshow(dog_image);
hold on

for i = 1:length(boxes)
   textStr = convertStringsToChars(class_labels(boxes(i).classIndex));
   position = [(boxes(i).cellIndex(2)-1)*448/7 (boxes(i).cellIndex(1)-1)*448/7];
   text(position(1),position(2),textStr,'Color',[0 1 0],'fontWeight','bold','fontSize',12);
   rectangle('Position',boxes(i).coords, 'EdgeColor','green','LineWidth',2);
end
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%%% Begin non max suppression. If intersection over union of any two bounding boxes is 
%%% higher than a defined threshold, and the two boxes contain the same class,
%%% remove the box with the the lower box probability.

for i = 1:length(boxes)
    for j = i+1:length(boxes)
        %%% calculate intersection over union (can also use bboxOverlapRatio
        %%% with proper toolbox
        intersect = rectint(boxes(i).coords,boxes(j).coords);
        union = boxes(i).coords(3)*boxes(i).coords(4)+boxes(j).coords(3)*boxes(j).coords(4)-intersect;
        iou(i,j) = intersect/union;
        if boxes(i).classIndex == boxes(j).classIndex && iou(i,j) > iou_thresh                
            [value(i) dropIndex(i)] = min([boxes(i).cellProb boxes(j).cellProb]);
            if dropIndex(i) == 1
                boxes(i).nonMax=0;
            elseif dropIndex(i) == 2
                boxes(j).nonMax=0;                
            end
        end                
    end
end

%%% plot result with non max suppression
figure(5) 
imshow(dog_image);
hold on
for i = 1:length(boxes)
    if boxes(i).nonMax == 1
      textStr = convertStringsToChars(class_labels(boxes(i).classIndex));
      position = [(boxes(i).cellIndex(2)-1)*448/7 (boxes(i).cellIndex(1)-1)*448/7];
      text(position(1),position(2),textStr,'Color',[0 1 0],'fontWeight','bold','fontSize',12);
      rectangle('Position',boxes(i).coords, 'EdgeColor','green','LineWidth',2);
    end
end

