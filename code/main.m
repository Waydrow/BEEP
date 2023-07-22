% BEEP & SimBEEP
clear,clc

global isIndependent;
global isTraining;
isIndependent = 1;
isTraining    = 0;

data = load('feature_5h.txt');
[line, col] = size(data); % dimensions
% 取前6个features, 依次为 S, V1, V2, A, T, C
ttt = data(:,1:6);
aaa = data(:,col);
% 包含最后一列，total sum of event count
data = [ttt,aaa];
% 训练集 : 测试集 = 4 : 1
x = floor(line*4/5);
trainX = data(1:x,:);
testX = data(x+1:end,:);
%%
% 变换 S, V1, V2
maxSum = max(data(:,1));
data1 = zeros(line, 3);
data1(:,1) = data(:,1) ./ (2 * maxSum);
data1(:,2) = data(:,2) .* 5 ./ (2 * maxSum);
data1(:,3) = data(:,3) .* 5 ./ (2 * maxSum);

trainX1 = data1(1:x,:);
testX1 = data1(x+1:end,:);

%%

% for inpendent situation
maxSum = max(trainX(:,1));
minSum = min(trainX(:,1));
% 归一化区间
ymin = 0.1;
ymax = 0.9;
for i = 1:x
    trainX(i,1) = (ymax - ymin) * (trainX(i,1) - minSum) / (maxSum - minSum) + ymin;
end
maxSum = max(testX(:,1));
minSum = min(testX(:,1));
for i = 1:line-x
    testX(i,1) = (ymax - ymin) * (testX(i,1) - minSum) / (maxSum - minSum) + ymin;
end



% hot event 阈值
% 1h: 7->30%, 8->25%, 10->20%, 12->15%, 18->10%(best)
threshold =7;
A = find(data(:,1) >= threshold); % get index of the hot events by the threshold
b = size(A,1);
fprintf('hot events / all = %f, under this threshold.\n', b/line);
Y = zeros(line,1);
Y(A) = 1;
trainY = Y(1:x,:);
testY = Y(x+1:end,:);
result = baseline0(trainX, trainX1, trainY, testX, testX1, testY);
% result = baseline01(trainX, trainX1, trainY, testX, testX1, testY);
% inputtest();

