%%
%Load Data
[XTrain,~,YTrain] = digitTrain4DArrayData;
[XValidation,~,YValidation] = digitTest4DArrayData;
dsTX = arrayDatastore(XTrain,'IterationDimension',4);
dsTY = arrayDatastore(YTrain);
dsTrain = combine(dsTX,dsTY);
numTrainImages = numel(YTrain);
figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(XTrain(:,:,:,idx(i)))
end
classes = categories(YTrain);
numClasses = numel(classes);
%%

%Check data normalization
figure
histogram(YTrain)
axis tight
ylabel('Counts')
xlabel('Rotation Angle')
%%
%Create newwork layers
layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    convolution2dLayer(3,8,'Padding','same','Name','c1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','rl1')
    averagePooling2dLayer(2,'Stride',2,'Name','p1')
    convolution2dLayer(3,16,'Padding','same','Name','c2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','rl2')
    averagePooling2dLayer(2,'Stride',2,'Name','p2')
    convolution2dLayer(3,32,'Padding','same','Name','c3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','rl3')
    convolution2dLayer(3,32,'Padding','same','Name','c4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','rl4')
    dropoutLayer(0.2,'Name','dl1')
    fullyConnectedLayer(1,'Name','cc1')];
lgraph = layerGraph(layers);

dlnet = dlnetwork(lgraph)
%%
%Train network
numEpochs = 30;
miniBatchSize = 128;
LearnRate = 1e-3;
learnRateDropPeriod = 20;
learnRateDropFactor = 0.1;

mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',@preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB',''});

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

velocity = [];

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX, dlY] = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        if mod(epoch,learnRateDropPeriod) == 0
            learnRate = learnRate * learnRateDropFactor;
        end
        
        % Update the network parameters using the SGDM optimizer.
        dlnet = dlupdate(randomlearningf,dlnet)
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

net.Layers
%%
%Test network
YPredicted = predict(net,XValidation);
predictionError = YValidation - YPredicted;

thr = 10;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(YValidation);

accuracy = numCorrect/numValidationImages
squares = predictionError.^2;
rmse = sqrt(mean(squares))
%%
%list the five worst predicted samples
[B,I] = maxk(predictionError,5)
figure
for i = 1:numel(I)
    subplot(1,5,i)    
    imshow(XValidation(:,:,:,idx(i)))
end
%%
%Visualize predictions
figure
scatter(YPredicted,YValidation,'+')
xlabel("Predicted Value")
ylabel("True Value")

hold on
plot([-60 60], [-60 60],'r--')
