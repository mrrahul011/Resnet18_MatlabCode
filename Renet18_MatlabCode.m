clc;
clear;
%%
%Input layer definition: 
input1 =[imageInputLayer([32 32 3],'Name','input'),
        convolution2dLayer(3,16,'Padding','same','Name','conv'),
        batchNormalizationLayer('Name','BN')];
%%
%Resnet18: 4 main(1to4) blocks with 2 sub blocks(A&B) in each block as defined below. 
%Each Sub block have 2 set of a coinvolution, batch
%normalization and a relu activation layer. So in 4 block there will be
%16 convolution layer in total and the input layer and output layer addes up to 18 layers. 

%Block 1
%Sub block 1A and 1B. 
block_1A = [
    convolution2dLayer(3,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')];

block_1B = [
    convolution2dLayer(3,16,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')];

%Block 2
%Sub block 2A and 2B
%Block 2A act as a maxpooling layer ie 3X3 convolution with stride of 2
block_2A = [
    convolution2dLayer(3,32,'Padding','same','Stride',2, 'Name','conv_5')
    batchNormalizationLayer('Name','BN_5')
    reluLayer('Name','relu_5')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_6')
    batchNormalizationLayer('Name','BN_6')
    reluLayer('Name','relu_6')];

block_2B = [
    convolution2dLayer(3,32,'Padding','same', 'Name','conv_7')
    batchNormalizationLayer('Name','BN_7')
    reluLayer('Name','relu_7')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_8')
    batchNormalizationLayer('Name','BN_8')
    reluLayer('Name','relu_8')];

%Block 3
%Sub block 3A and 3B
%Block 3A act as a maxpooling layer ie 3X3 convolution with stride of 2
block_3A = [
    convolution2dLayer(3,64,'Padding','same','Stride',2, 'Name','conv_9')
    batchNormalizationLayer('Name','BN_9')
    reluLayer('Name','relu_9')
    
    convolution2dLayer(3,64,'Padding','same','Name','conv_10')
    batchNormalizationLayer('Name','BN_10')
    reluLayer('Name','relu_10')];

block_3B = [
    convolution2dLayer(3,64,'Padding','same', 'Name','conv_11')
    batchNormalizationLayer('Name','BN_11')
    reluLayer('Name','relu_11')
    
    convolution2dLayer(3,64,'Padding','same','Name','conv_12')
    batchNormalizationLayer('Name','BN_12')
    reluLayer('Name','relu_12')];

%Block 4
%Sub block 4A and 4B
%Block 4A act as a maxpooling layer ie 3X3 convolution with stride of 2
block_4A = [
    convolution2dLayer(3,128,'Padding','same','Stride',2, 'Name','conv_13')
    batchNormalizationLayer('Name','BN_13')
    reluLayer('Name','relu_13')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv_14')
    batchNormalizationLayer('Name','BN_14')
    reluLayer('Name','relu_14')];

block_4B = [
    convolution2dLayer(3,128,'Padding','same', 'Name','conv_15')
    batchNormalizationLayer('Name','BN_15')
    reluLayer('Name','relu_15')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv_16')
    batchNormalizationLayer('Name','BN_16')
    reluLayer('Name','relu_16')];

%%

%1X1 Convolution with stride of 2 for Skip connection layer
connect1 = convolution2dLayer(1,16,'Padding','same','Stride',2, 'Name','skipconv_1');

connect2 = convolution2dLayer(1,32,'Padding','same','Stride',2, 'Name','skipconv_2');

connect3 = convolution2dLayer(1,64,'Padding','same','Stride',2, 'Name','skipconv_3');

%%
%output layer definition
outlayer = [    
    convolution2dLayer(1,10,'Padding','same', 'Name','conv_1x1')

    fullyConnectedLayer(10,'Name','fc1');

    softmaxLayer('Name','sm1');

    classificationLayer('Name','output')];
%%
%Definition of addition layer to combine two connection
add1 = additionLayer(2,'Name','add_1');
add2 = additionLayer(2,'Name','add_2');
add3 = additionLayer(2,'Name','add_3');
add4 = additionLayer(2,'Name','add_4');
add5 = additionLayer(2,'Name','add_5');
add6 = additionLayer(2,'Name','add_6');
add7 = additionLayer(2,'Name','add_7');
add8 = additionLayer(2,'Name','add_8');
%%
%Layer definition for Resnet 18
%{
layerGraph:A layer graph specifies the architecture of a deep learning 
network with a more complex graph structure in which layers can have 
inputs from multiple layers and outputs to multiple layers 
%}
%addLayers: add layer to the network
%connectLayers: connects the source layer to the destination layer.
lgraph = layerGraph;

lgraph = addLayers(lgraph,input1);
lgraph = addLayers(lgraph,block_1A);
lgraph = addLayers(lgraph,add1);
lgraph = connectLayers(lgraph,'BN','conv_1');
lgraph = connectLayers(lgraph,'BN','add_1/in1');
lgraph = connectLayers(lgraph,'relu_2','add_1/in2');

lgraph = addLayers(lgraph,block_1B);
lgraph = addLayers(lgraph,add2);
lgraph = connectLayers(lgraph,'add_1','conv_3');
lgraph = connectLayers(lgraph,'relu_4','add_2/in1');
lgraph = connectLayers(lgraph,'add_1','add_2/in2');

lgraph = addLayers(lgraph,block_2A);
lgraph = addLayers(lgraph,add3);
lgraph = addLayers(lgraph,connect1);
lgraph = connectLayers(lgraph,'add_2','conv_5');
lgraph = connectLayers(lgraph,'add_2','skipconv_1');
lgraph = connectLayers(lgraph,'relu_6','add_3/in1');
lgraph = connectLayers(lgraph,'skipconv_1','add_3/in2');

lgraph = addLayers(lgraph,block_2B);
lgraph = addLayers(lgraph,add4);
lgraph = connectLayers(lgraph,'add_3','conv_7');
lgraph = connectLayers(lgraph,'relu_8','add_4/in1');
lgraph = connectLayers(lgraph,'add_3','add_4/in2');
% 
lgraph = addLayers(lgraph,block_3A);
lgraph = addLayers(lgraph,add5);
lgraph = addLayers(lgraph,connect2);
lgraph = connectLayers(lgraph,'add_4','conv_9');
lgraph = connectLayers(lgraph,'add_4','skipconv_2');
lgraph = connectLayers(lgraph,'relu_10','add_5/in1');
lgraph = connectLayers(lgraph,'skipconv_2','add_5/in2');
% % 
lgraph = addLayers(lgraph,block_3B);
lgraph = addLayers(lgraph,add6);
lgraph = connectLayers(lgraph,'add_5','conv_11');
lgraph = connectLayers(lgraph,'relu_12','add_6/in1');
lgraph = connectLayers(lgraph,'add_5','add_6/in2');
% 
lgraph = addLayers(lgraph,block_4A);
lgraph = addLayers(lgraph,add7);
lgraph = addLayers(lgraph,connect3);
lgraph = connectLayers(lgraph,'add_6','conv_13');
lgraph = connectLayers(lgraph,'add_6','skipconv_3');
lgraph = connectLayers(lgraph,'relu_14','add_7/in1');
lgraph = connectLayers(lgraph,'skipconv_3','add_7/in2');
% 
lgraph = addLayers(lgraph,block_4B);
lgraph = addLayers(lgraph,add8);
lgraph = connectLayers(lgraph,'add_7','conv_15');
lgraph = connectLayers(lgraph,'relu_16','add_8/in1');
lgraph = connectLayers(lgraph,'add_7','add_8/in2');
% 
lgraph = addLayers(lgraph,outlayer);
lgraph = connectLayers(lgraph,'add_8','conv_1x1');

%To visualise the entire network use plot command.
figure
plot(lgraph)
%%

%Download CIFAR10 data set from matlab website.
rootFolder = 'cifar10Train';
%Create an imagedata store with label name as the subfolder name.
imds_train = imageDatastore(fullfile(rootFolder), ...
             'IncludeSubfolders',true, 'LabelSource', 'foldernames');
%resize the image
% imds_train.ReadSize = numpartitions(imds_train);
% imds_train.ReadFcn = @(loc)imresize(imread(loc),[256,256]);

%%
% Load Test Data

rootFolder = 'cifar10Test';
imds_test = imageDatastore(fullfile(rootFolder), ...
    	      'IncludeSubfolders',true, 'LabelSource', 'foldernames');     
% %resize the image
% imds_test.ReadSize = numpartitions(imds_test);
% imds_test.ReadFcn = @(loc)imresize(imread(loc),[256,256]);
%%
%Training options. 
%set checkpoint path to the current working directory. This will hen
%resuming the training if interupted.
checkpointPath = pwd;
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs',5, ...
    'MiniBatchSize', 32, ...
    'Verbose',0, ...
    'ValidationData',imds_test, ...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress',...
    'CheckpointPath',checkpointPath);
%{
additional training options available in matlab
  'Shuffle','every-epoch',...
  'LearnRateSchedule', 'piecewise', ...
  'ValidationData',{imds_test}, ...
  'ValidationFrequency',5, ...
  'LearnRateDropFactor', 0.1, ...
  'LearnRateDropPeriod', 8, ...
  'L2Regularization', 0.004, ...

%}
%%
% Train network
[net, info] = trainNetwork(imds_train, lgraph, options);
resnet18v1 = net;
save resnet18v1;

%%
%{
step to load Load Checkpoint and restart training.

Load checkpoint and train again
load('net_checkpoint__195__2018_07_13__11_59_10.mat','net');

opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', ...
    'Verbose', true, ...
    'ValidationData',imds_test, ...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress',...
    'CheckpointPath',checkpointPath);;

Train
[net, info] = trainNetwork(imds_train, layerGraph(net), opts);
%}
%%

% classify validation images and compute accuracy
YPred = classify(resnet,imds_test);
YValidation = imds_test.Labels;

%%
%calculate accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)
%%
% Testing the trained model 
labels = classify(resnet, imds_test);
ii = randi(21);
im = imread(imds_test.Files{ii});
figure, imshow(im);
if labels(ii) == imds_test.Labels(ii)
   colorText = 'g'; 
else
    colorText = 'r';
end
title(char(labels(ii)),'Color',colorText);
%%