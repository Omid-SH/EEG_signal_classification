%% Load and Store Data (0)
clear; clc

load('CI_Project_data.mat');
nTrain = 165;
nTest = 45;
nChannel = 30;

Train = struct;
for i = 1:nTrain
    Train(i).Data = TrainData(:,:,i);
    Train(i).Label = TrainLabel(i);
end

Test = struct;
for i = 1:nTest
    Test(i).Data = TestData(:,:,i);
end
%% Feature mining (1-A)

%% statistical features extraction
%MAXX = zeros(1, 5);
for i = 1:nTrain
    for channel = 1:nChannel
        % Variance
        Train(i).Var(channel).Var = var(Train(i).Data(channel,:));
        %MAXX(1) = max(abs(var(Train(i).Data(channel,:))), MAXX(1));
        
        % Domain Histogram >-10, [-10,-8], [-8,-6], ... , [8 10], >10
        L = length(Train(i).Data(channel,:));
        P = [sum(Train(i).Data(channel,:) > 10)];
        for r=1:10
            P = [P sum(Train(i).Data(channel,:) > 10-2*r)-P(end)];
        end
        P = [P L-P(end)];
        P = P ./ L;
        Train(i).DH(channel).DH = P;
        %MAXX(2) = max(max(abs(P)), MAXX(2));
        
        % AR model's coefficients
        S = ar(Train(i).Data(channel,:),6);
        Train(i).AR(channel).Coef = S.A;
        %MAXX(3) = max(max(abs(S.A)), MAXX(3));
        
        % Form Factor
        Train(i).FF(channel).FF = rms(Train(i).Data(channel,:)) ./ mean(Train(i).Data(channel,:));
        %MAXX(4) = max(abs(Train(i).FF(channel).FF), MAXX(4));
        
        % Xcorr
        C = zeros(1,nChannel);
        for c = 1:nChannel
            C(c) = xcorr(Train(i).Data(channel,:), Train(i).Data(c,:), 0, 'coeff');
        end
        Train(i).XC(channel).XC = C; 
        %MAXX(5) = max(max(abs(C)), MAXX(5));
    end
end

for i = 1:nTest
    for channel = 1:nChannel
        % Variance
        Test(i).Var(channel).Var = var(Test(i).Data(channel,:));
        %MAXX(1) = max(abs(var(Train(i).Data(channel,:))), MAXX(1));
        
        % Domain Histogram >-10, [-10,-8], [-8,-6], ... , [8 10], >10
        L = length(Test(i).Data(channel,:));
        P = [sum(Test(i).Data(channel,:) > 10)];
        for r=1:10
            P = [P sum(Test(i).Data(channel,:) > 10-2*r)-P(end)];
        end
        P = [P L-P(end)];
        P = P ./ L;
        Test(i).DH(channel).DH = P;
        %MAXX(2) = max(max(abs(P)), MAXX(2));
        
        % AR model's coefficients
        S = ar(Test(i).Data(channel,:),6);
        Test(i).AR(channel).Coef = S.A;
        %MAXX(3) = max(max(abs(S.A)), MAXX(3));
        
        % Form Factor
        Test(i).FF(channel).FF = rms(Test(i).Data(channel,:)) ./ mean(Test(i).Data(channel,:));
        %MAXX(4) = max(abs(Train(i).FF(channel).FF), MAXX(4));
        
        % Xcorr
        C = zeros(1,nChannel);
        for c = 1:nChannel
            C(c) = xcorr(Test(i).Data(channel,:), Test(i).Data(c,:), 0, 'coeff');
        end
        Test(i).XC(channel).XC = C; 
        %MAXX(5) = max(max(abs(C)), MAXX(5));
    end
end

% Normalize
% for i = 1:nTrain
%     for channel = 1:nChannel
%         % Variance
%         Train(i).Var(channel).Var = Train(i).Var(channel).Var ./ MAXX(1);
%         
%         % Domain Histogram 
%         Train(i).DH(channel).DH = Train(i).DH(channel).DH ./ MAXX(2);
%         
%         % AR model's coefficients
%         S = ar(Train(i).Data(channel,:),6);
%         Train(i).AR(channel).Coef = Train(i).AR(channel).Coef ./ MAXX(3);
%         
%         % Form Factor
%         Train(i).FF(channel).FF = Train(i).FF(channel).FF ./ MAXX(4);
%         
%         % Xcorr
%         Train(i).XC(channel).XC = Train(i).XC(channel).XC ./ MAXX(5); 
%     end
% end

%% frequency features extraction

Fs = 256;

for i = 1:nTrain
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        FFT = abs(fft(Train(i).Data(channel,:)));
        F = find(FFT == max(FFT));
        MaxF = F(1)./ (length(Train(i).Data(channel,:))-1) .* Fs;
        
        AvgF = meanfreq(Train(i).Data(channel,:), Fs);
        
        MedF = medfreq(Train(i).Data(channel,:), Fs);
        
        Train(i).SF(channel).SF = [MaxF AvgF MedF];
        
        % Band Relative Power
        pRMS = rms(Train(i).Data(channel,:))^2;
        B = zeros(1, 7);
        B(1) = bandpower(Train(i).Data(channel,:),Fs,[2 8]);
        B(2) = bandpower(Train(i).Data(channel,:),Fs,[9 15]);
        B(3) = bandpower(Train(i).Data(channel,:),Fs,[16 22]);
        B(4) = bandpower(Train(i).Data(channel,:),Fs,[23 29]);
        B(5) = bandpower(Train(i).Data(channel,:),Fs,[30 36]);
        B(6) = bandpower(Train(i).Data(channel,:),Fs,[37 43]);
        B(7) = bandpower(Train(i).Data(channel,:),Fs,[44 50]);
        B = B ./ pRMS;
        Train(i).BP(channel).BP = B; 
        Train(i).PW(channel).PW = pRMS; 
        
        % Entropy
        Train(i).EN(channel).EN = entropy(Train(i).Data(channel,:));
    end
end

for i = 1:nTest
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        FFT = abs(fft(Test(i).Data(channel,:)));
        F = find(FFT == max(FFT));
        MaxF = F(1)./ (length(Test(i).Data(channel,:))-1) .* Fs;
        
        AvgF = meanfreq(Test(i).Data(channel,:), Fs);
        
        MedF = medfreq(Test(i).Data(channel,:), Fs);
        
        Test(i).SF(channel).SF = [MaxF AvgF MedF];
        
        % Band Relative Power
        pRMS = rms(Test(i).Data(channel,:))^2;
        B = zeros(1, 7);
        B(1) = bandpower(Test(i).Data(channel,:),Fs,[2 8]);
        B(2) = bandpower(Test(i).Data(channel,:),Fs,[9 15]);
        B(3) = bandpower(Test(i).Data(channel,:),Fs,[16 22]);
        B(4) = bandpower(Test(i).Data(channel,:),Fs,[23 29]);
        B(5) = bandpower(Test(i).Data(channel,:),Fs,[30 36]);
        B(6) = bandpower(Test(i).Data(channel,:),Fs,[37 43]);
        B(7) = bandpower(Test(i).Data(channel,:),Fs,[44 50]);
        B = B ./ pRMS;
        Test(i).BP(channel).BP = B; 
        Test(i).PW(channel).PW = pRMS; 
        
        % Entropy
        Test(i).EN(channel).EN = entropy(Test(i).Data(channel,:));
    end
end

%% Make Final feature Vector

% Train
% Statistical features
% 30 + 30*12 + 30*7 + 30 + 30*30
for i = 1:nTrain
    Train(i).sfeatures = zeros(1, 1530);
end
% Frequency features
% 30*3 + 30*7 + 30 + 30
for i = 1:nTrain
    Train(i).ffeatures = zeros(1, 360);
end

pointer = 1;
for i = 1:nTrain
    pointer = 1;
    for channel = 1:nChannel
        % Variance
        Train(i).sfeatures(pointer) = Train(i).Var(channel).Var;
        pointer = pointer + 1;
    end
end

for i = 1:nTrain
    pointer = 30 + 1;
    for channel = 1:nChannel
        % Domain Histogram 
        Train(i).sfeatures(pointer:pointer+11) = Train(i).DH(channel).DH;
        pointer = pointer + 12;
    end
end

for i = 1:nTrain
    pointer = 30 + 30*12 + 1;
    for channel = 1:nChannel
        % AR model's coefficients
        Train(i).sfeatures(pointer:pointer+6) = Train(i).AR(channel).Coef;
        pointer = pointer + 7;
    end
end

for i = 1:nTrain
    pointer = 30 + 30*12 + 30*7 + 1;
    for channel = 1:nChannel
        % Form Factor
        Train(i).sfeatures(pointer) = Train(i).FF(channel).FF;
        pointer = pointer + 1;
    end
end

for i = 1:nTrain
    pointer = 30 + 30*12 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Xcorr
        Train(i).sfeatures(pointer:pointer+29) = Train(i).XC(channel).XC;
        pointer = pointer + 30;
    end
end       

for i = 1:nTrain
    pointer = 30 + 30*12 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Xcorr
        Train(i).sfeatures(pointer:pointer+29) = Train(i).XC(channel).XC;
        pointer = pointer + 30;
    end
end 

% Freq

for i = 1:nTrain
    pointer = 1;
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        Train(i).ffeatures(pointer:pointer+2) = Train(i).SF(channel).SF;
        pointer = pointer + 3;
    end
end 

for i = 1:nTrain
    pointer = 30*3 + 1;
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        Train(i).ffeatures(pointer:pointer+6) = Train(i).BP(channel).BP;
        pointer = pointer + 7;
    end
end 

for i = 1:nTrain
    pointer = 30*3 + 30*7 + 1;
    for channel = 1:nChannel
        % Signal Power
        Train(i).ffeatures(pointer) = Train(i).PW(channel).PW;
        pointer = pointer + 1;
    end
end

for i = 1:nTrain
    pointer = 30*3 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Signal Entropy
        Train(i).ffeatures(pointer) = Train(i).EN(channel).EN;
        pointer = pointer + 1;
    end
end 


% Test
% Statistical features
% 30 + 30*12 + 30*7 + 30 + 30*30
for i = 1:nTest
    Test(i).sfeatures = zeros(1, 1530);
end
% Frequency features
% 30*3 + 30*7 + 30 + 30
for i = 1:nTest
    Test(i).ffeatures = zeros(1, 360);
end

pointer = 1;
for i = 1:nTest
    pointer = 1;
    for channel = 1:nChannel
        % Variance
        Test(i).sfeatures(pointer) = Test(i).Var(channel).Var;
        pointer = pointer + 1;
    end
end

for i = 1:nTest
    pointer = 30 + 1;
    for channel = 1:nChannel
        % Domain Histogram 
        Test(i).sfeatures(pointer:pointer+11) = Test(i).DH(channel).DH;
        pointer = pointer + 12;
    end
end

for i = 1:nTest
    pointer = 30 + 30*12 + 1;
    for channel = 1:nChannel
        % AR model's coefficients
        Test(i).sfeatures(pointer:pointer+6) = Test(i).AR(channel).Coef;
        pointer = pointer + 7;
    end
end

for i = 1:nTest
    pointer = 30 + 30*12 + 30*7 + 1;
    for channel = 1:nChannel
        % Form Factor
        Test(i).sfeatures(pointer) = Test(i).FF(channel).FF;
        pointer = pointer + 1;
    end
end

for i = 1:nTest
    pointer = 30 + 30*12 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Xcorr
        Test(i).sfeatures(pointer:pointer+29) = Test(i).XC(channel).XC;
        pointer = pointer + 30;
    end
end       

for i = 1:nTest
    pointer = 30 + 30*12 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Xcorr
        Test(i).sfeatures(pointer:pointer+29) = Test(i).XC(channel).XC;
        pointer = pointer + 30;
    end
end 

% Freq

for i = 1:nTest
    pointer = 1;
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        Test(i).ffeatures(pointer:pointer+2) = Test(i).SF(channel).SF;
        pointer = pointer + 3;
    end
end 

for i = 1:nTest
    pointer = 30*3 + 1;
    for channel = 1:nChannel
        % Max, AVG and Median Freq of a signal
        Test(i).ffeatures(pointer:pointer+6) = Test(i).BP(channel).BP;
        pointer = pointer + 7;
    end
end 

for i = 1:nTest
    pointer = 30*3 + 30*7 + 1;
    for channel = 1:nChannel
        % Signal Power
        Test(i).ffeatures(pointer) = Test(i).PW(channel).PW;
        pointer = pointer + 1;
    end
end

for i = 1:nTest
    pointer = 30*3 + 30*7 + 30 + 1;
    for channel = 1:nChannel
        % Signal Entropy
        Test(i).ffeatures(pointer) = Test(i).EN(channel).EN;
        pointer = pointer + 1;
    end
end 

%% Feature selection (1-B) [Select 10 features. 4 Best Freq and 6 others]

%% Make FeatureSelection Struct
% Statistical features
% 30 + 30*12 + 30*7 + 30 + 30*30
FeatureSelection.sFeatureMatrix = zeros(1530, nTrain);
for i = 1:nTrain  
    FeatureSelection.sFeatureMatrix(:, i) = Train(i).sfeatures';
end

% Calculate Fisher Score of each feature
FeatureSelection.Fisher_sFeatureMatrix = zeros(1530, 1);
Labels = zeros(1,nTrain);
for i = 1:nTrain  
    Labels(i) = Train(i).Label;
end

for i = 1:1530  
    FeatureSelection.Fisher_sFeatureMatrix(i) = FisherC1(FeatureSelection.sFeatureMatrix(i, :), Labels);
end


% Frequency features
% 30*3 + 30*7 + 30 + 30
FeatureSelection.fFeatureMatrix = zeros(360, nTrain);
for i = 1:nTrain
    FeatureSelection.fFeatureMatrix(:, i) = Train(i).ffeatures';
end

% Calculate Fisher Score of each feature
FeatureSelection.Fisher_fFeatureMatrix = zeros(360, 1);

for i = 1:360  
    FeatureSelection.Fisher_fFeatureMatrix(i) = FisherC1(FeatureSelection.fFeatureMatrix(i, :), Labels);
end

% Make feature Matrix for Test
% Statistical features
% 30 + 30*12 + 30*7 + 30 + 30*30
TestFeature.sFeatureMatrix = zeros(1530, nTest);
for i = 1:nTest  
    TestFeature.sFeatureMatrix(:, i) = Test(i).sfeatures';
end

% Frequency features
% 30*3 + 30*7 + 30 + 30
TestFeature.fFeatureMatrix = zeros(360, nTest);
for i = 1:nTest
    TestFeature.fFeatureMatrix(:, i) = Test(i).ffeatures';
end

%% Find Selected features
%%
[Fisher_ssorted, Fisher_sindex] = sort(FeatureSelection.Fisher_sFeatureMatrix);
[Fisher_fsorted, Fisher_findex] = sort(FeatureSelection.Fisher_fFeatureMatrix);

%% Select and make Train Data and Final Test data and Normalize them 
New_Train_Data = [];
New_Test_Data =[];
New_Train_Data = [New_Train_Data FeatureSelection.sFeatureMatrix(Fisher_sindex([1483 1486 1493 1497 1499 1500]),:)]; % 6 from statistical
New_Train_Data = [New_Train_Data ; FeatureSelection.fFeatureMatrix(Fisher_findex([283 318 324 352]),:)]; % 4 from frequeny

%New_Train_Data = [New_Train_Data FeatureSelection.sFeatureMatrix(Fisher_sindex(1300:1500),:)]; % 6 from statistical
%New_Train_Data = [New_Train_Data ; FeatureSelection.fFeatureMatrix(Fisher_findex(300:360),:)]; % 4 from frequeny

New_Test_Data = [New_Test_Data TestFeature.sFeatureMatrix(Fisher_sindex([1483 1486 1493 1497 1499 1500]),:)]; % 6 from statistical
New_Test_Data = [New_Test_Data ; TestFeature.fFeatureMatrix(Fisher_findex([283 318 324 352]),:)]; % 4 from frequeny

% Normalize
[New_Train_Data, PS] = mapminmax(New_Train_Data);
New_Test_Data = mapminmax('apply',New_Test_Data,PS);

% Check if variance is acceptable 
% hist(New_Train_Data(8,:))

New_Train_Data_Label = Labels;

%% MLP training using different feature sets with 5-fold-cross-validation
% (1-C)

%% 1 Hidden Layer
%Neurons = [5 10 20 30 40 60 80 100 165];
Neurons = [5:15];

acc_max = 0;
best_param = [];

for neuron = Neurons
    
    ACC = 0;
    
    % 5-fold cross-validation
    K = 5;
    for k = 1:5
        net = patternnet(neuron);
        % Setup Division of Data for Training, Validation, Testing. We do not
        % need and validation and test by train function
%         net.divideParam.trainRatio = 0.8;
%         net.divideParam.valRatio = 0.2;
%         net.divideParam.testRatio = 0;
%         net.trainParam.epochs = 10000;
    
        % Select Tain and Test Data
        Train_temp = New_Train_Data(:, [1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
        Train_temp_Label = New_Train_Data_Label([1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
        Test_temp = New_Train_Data(:, [((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);
        Test_temp_Label = New_Train_Data_Label([((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);

        [net,tr] = train(net, Train_temp, Train_temp_Label);
                 
        Y_o = net(Test_temp) > 0.5;
        ACC = ACC + sum(Y_o == Test_temp_Label);
        
    end
   
    ACC = ACC ./ nTrain;
        
    % Find best parameters for model
    if(ACC > acc_max)
        acc_max = ACC;
        best_param = neuron;
    end
end

fprintf('Best accuracy for 1 hidden layer using 5-fold cross-validation is %f%% with %i neurons in hidden layer. \n',acc_max*100, best_param);

%% 2 Hidden Layers

Neurons = [5 10 20 30 40 60 80 100 165];
%Neurons = [50:70];
%Neurons = [75:85];
Neurons = [55:65];

acc_max = 0;
best_param = [];

for neuron = Neurons
    
    ACC = 0;
    
    % 5-fold cross-validation
    K = 5;
    for k = 1:5
        net = patternnet([neuron neuron]);
        % Setup Division of Data for Training, Validation, Testing. We do not
        % need and validation and test by train function
         net.divideParam.trainRatio = 0.9;
         net.divideParam.valRatio = 0.1;
         %net.divideParam.testRatio = 0;

    
        % Select Tain and Test Data
        Train_temp = New_Train_Data(:, [1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
        Train_temp_Label = New_Train_Data_Label([1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
        Test_temp = New_Train_Data(:, [((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);
        Test_temp_Label = New_Train_Data_Label([((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);

        [net,tr] = train(net, Train_temp, Train_temp_Label);
                 
        Y_o = round(net(Test_temp)) > 0.5;
        ACC = ACC + sum(Y_o == Test_temp_Label);
        
    end
   
    ACC = ACC ./ nTrain;
        
    % Find best parameters for model
    if(ACC > acc_max)
        acc_max = ACC;
        best_param = neuron;
    end
    
end

fprintf('Best accuracy for 2 similar hidden layer using 5-fold cross-validation is %f%% with %i neurons in each hidden layer. \n',acc_max*100, best_param);

%% RBF training using different feature sets with 5-fold-cross-validation
% (1-D)

%Spreads = [0.01, 0.1, 1, 10];
%MNs = [5, 10, 20, 40, 60, 80, 100, 165];
%Spreads = [8, 10, 12, 14];
%MNs = [2, 4, 6, 8];
%Spreads = [5, 6, 7, 8, 9];
%MNs = [5, 6, 7, 8, 9, 10];
%Spreads = [6.7, 6.8, 6.9, 7, 7.1, 7.2, 7.3];
%MNs = 8;

Spreads = [1 2 3 4];
MNs = [1 2 3 4 5];

acc_max = 0;
best_param = [];

for spread = Spreads
    for MN = MNs
        ACC = 0;

        % 5-fold cross-validation
        K = 5;
        for k = 1:5
            % Select Tain and Test Data
            Train_temp = New_Train_Data(:, [1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
            Train_temp_Label = New_Train_Data_Label([1:((k-1)*(nTrain/K)) ((k)*(nTrain/K)+1):nTrain]);
            Test_temp = New_Train_Data(:, [((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);
            Test_temp_Label = New_Train_Data_Label([((k-1)*(nTrain/K)+1):((k)*(nTrain/K))]);
            net = newrb(Train_temp, Train_temp_Label, 0, spread, MN);

            Y_o = sim(net, Test_temp) > 0.5;
            ACC = ACC + sum(Y_o == Test_temp_Label);

        end

        ACC = ACC ./ nTrain;

        % Find best parameters for model
        if(ACC > acc_max)
            acc_max = ACC;
            best_param = [spread MN];
        end
    end
end

fprintf('Best accuracy for RBF network using 5-fold cross-validation is %f%% with Spread = %f and MNS = %i \n',acc_max*100, best_param(1), best_param(2));

%% Best networks output for validation data (1-F)

%% MLP 1 Hidden Layer
%neuron = 55;
%neuron = 165;
neuron = 9;

net = patternnet(neuron);
[net,tr] = train(net, New_Train_Data, New_Train_Data_Label);

%Test_Label_MLP_1_1 = net(New_Test_Data) > 0.5;
%save('Test_Label_MLP_1_1.mat','Test_Label_MLP_1_1');

%Test_Label_MLP_1_2_1 = net(New_Test_Data) > 0.5;
%save('Test_Label_MLP_1_2_1.mat','Test_Label_MLP_1_2_1');

Test_Label_MLP_1_2_2 = net(New_Test_Data) > 0.5;
save('Test_Label_MLP_1_2_2.mat','Test_Label_MLP_1_2_2');

%% MLP 2 Hidden Layers
%neuron = 51;
%neuron = 81; 
neuron = 64;

net = patternnet([neuron neuron]);
[net,tr] = train(net, New_Train_Data, New_Train_Data_Label);


%Test_Label_MLP_2_1 = net(New_Test_Data) > 0.5;
%save('Test_Label_MLP_2_1.mat','Test_Label_MLP_2_1');

%Test_Label_MLP_2_2_1 = net(New_Test_Data) > 0.5;
%save('Test_Label_MLP_2_2_1.mat','Test_Label_MLP_2_2_1');

Test_Label_MLP_2_2_2 = net(New_Test_Data) > 0.5;
save('Test_Label_MLP_2_2_2.mat','Test_Label_MLP_2_2_2');
%% RBF training using different feature sets with 5-fold-cross-validation

% Spread = 6.7;
% MN = 8;
%Spread = 0.9;
%MN = 12;
Spread = 3;
MN = 3;

net = newrb(New_Train_Data, New_Train_Data_Label, 0, spread, MN);

%Test_Label_RBF_1 = sim(net, Test_temp) > 0.5;
%save('Test_Label_RBF_1.mat','Test_Label_RBF_1');

%Test_Label_RBF_2_1 = sim(net, Test_temp) > 0.5;
%save('Test_Label_RBF_2_1.mat','Test_Label_RBF_2_1');

Test_Label_RBF_2_2 = sim(net, Test_temp) > 0.5;
save('Test_Label_RBF_2_2.mat','Test_Label_RBF_2_2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Phase 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Feature selection using genetic algorithm (2-A-1)

options = optimoptions('ga');
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'PlotFcn', {  @gaplotbestf @gaplotscorediversity @gaplotscores });

[result, fval] = ga(@fitness_function,10,[],[],[],[],ones(1,10),1890.*ones(1,10),[],1:10,options);

%%
New_Train_Data = zeros(length(result), nTrain);
New_Test_Data = zeros(length(result), nTrain);

for i = 1:length(result)
    if (result(i) <= 1530)
        New_Train_Data(i, :) = FeatureSelection.sFeatureMatrix(result(i),:);
        New_Test_Data(i, :) = FeatureSelection.sFeatureMatrix(result(i),:);
    else
        New_Train_Data(i, :) = FeatureSelection.fFeatureMatrix(result(i)-1530,:);
        New_Test_Data(i, :) = FeatureSelection.fFeatureMatrix(result(i)-1530,:);
    end
end

% Normalize
[New_Train_Data, PS] = mapminmax(New_Train_Data);
New_Test_Data = mapminmax('apply',New_Test_Data,PS);
New_Train_Data_Label = Labels;

%% Feature selection using PSO algorithm (2-A-2)
options = optimoptions('particleswarm');
options = optimoptions(options,'PlotFcn', {  @pswplotbestf });
[result, fval] = particleswarm(@fitness_function_pso, 10, 1.*ones(1,10), 1890.*ones(1,10), options);

%%
result = round(result);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function J = FisherC1 (x, L)
    % Finsher Score Calculator for 1-D
    index_b = (L == 0);
    x0 = x(index_b);
    index_b = (L == 1);
    x1 = x(index_b);
    
    mu0 = mean(x);
    mu1 = mean(x0);
    mu2 = mean(x1);
    var1 = var(x0);
    var2 = var(x1);
    
    J = ((mu0-mu1)^2 + (mu0-mu2)^2) ./ (var1+var2);
end

function J = FisherC2 (x, L)
    % Finsher Score Calculator for p-D
    p = size(x, 1);
    
    index_b = (L == 0);
    x0 = x(:, index_b);
    index_b = (L == 1);
    x1 = x(:, index_b);
    
    mu0 = mean(x, 2);
    mu1 = mean(x0, 2);
    mu2 = mean(x1, 2);   
    Sb = (mu1-mu0)*(mu1-mu0)' + (mu2-mu0)*(mu2-mu0)';
    
    S1 = zeros(p);
    S2 = zeros(p);
    for i=1:size(x, 2)
        if(index_b(i))
            S2 = S2 + (x(:,i)-mu2)*(x(:,i)-mu2)';
        else
            S1 = S1 + (x(:,i)-mu1)*(x(:,i)-mu1)';
        end
    end
    
    Sw = S1 ./ size(x0, 2) + S2 ./ size(x1, 2);
    
    J = trace(Sb) ./ trace(Sw);
end

function cost = fitness_function(in)
    
    % Load data
    Load = load('All.mat');
    nTrain = Load.nTrain;
    F1 = Load.FeatureSelection.sFeatureMatrix;
    F2 = Load.FeatureSelection.fFeatureMatrix;
    
    X = zeros(length(in), nTrain);
    for i = 1:length(in)
        if (in(i) <= 1530)
            X(i,:) = F1(in(i), :);
        elseif (in(i) <= 1890)
            X(i,:) = F2(in(i)-1530, :);
        else
            cost = inf;
            return;
        end
    end
    
    cost = FisherC2 (X, Load.New_Train_Data_Label);
    if (cost ~= inf)
        cost = -cost;
    end
end

function cost = fitness_function_pso(in)
    
    % Load data
    Load = load('All.mat');
    nTrain = Load.nTrain;
    F1 = Load.FeatureSelection.sFeatureMatrix;
    F2 = Load.FeatureSelection.fFeatureMatrix;
    
    in = round(in);
    X = zeros(length(in), nTrain);
    for i = 1:length(in)
        if (in(i) <= 1530)
            X(i,:) = F1(in(i), :);
        elseif (in(i) <= 1890)
            X(i,:) = F2(in(i)-1530, :);
        else
            cost = inf;
            return;
        end
    end
    
    cost = FisherC2 (X, Load.New_Train_Data_Label);
    if (cost ~= inf)
        cost = -cost;
    end
end

