load fisheriris.mat; % Load dataset

species = grp2idx(species); % Creates index vector 

index1 = randperm(150,60)'; % Select indexes for testing randomly 
index2 = setdiff((1:150)',index1); % Select indexes for training 
testingData = meas(index1,:); % Create testing data
testingTarget = species(index1,:); % Create testing target 
trainingData = meas(index2,:); % Create traning data 
trainingTarget = species(index2,:); % Create traning targrt 

hiddenLayerPerformance_array = []; % Initilize array for average performance of each hidden layer

for hiddenLayers = 5: +5: 20 % Iteration for 4 hidden layer sizes [5,10,15,20]

    testPerformance_array = []; % Initialize array for performace of each iteration 

   for n = 1 : +1 : 10 % Iteration for 10 times experiment
      
      net = feedforwardnet(hiddenLayers); % Construct the feedforward network 
      [net,tr]=train(net,trainingData',trainingTarget'); % Train the network
      %view(net); Line 32 is the specific line for vewing the network 
      testingOutput = net(testingData'); % Estimate the targets using the trained network 
      testPerformace = length(find(round(testingOutput)-testingTarget'==0))*100/length(testingOutput); % Calculate the acuuracy of iertation in ecah hidden layer size
      testPerformance_array(end+1) = testPerformace; % Append the performance of each iteration into array 

   end 

   view(net); % View the netwok 
   avg_testPerformace = mean(testPerformance_array); % Calculate average performace of each hidden layer size 
   hiddenLayerPerformance_array(end+1) = avg_testPerformace; % Append the performace of each hidden layer size into array 
   
end

