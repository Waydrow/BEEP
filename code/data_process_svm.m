function y = data_process_svm(data, threshold, istrain)

[line, ~] = size(data);
x = floor(line*4/5);
trainX = data(1:x,:);
testX = data(x+1:end,:);
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

A = find(data(:,1) >= threshold); % get index of the hot events by the threshold
b = size(A,1);
Y = zeros(line,1);
Y(A) = 1;
trainY = Y(1:x,:);
testY = Y(x+1:end,:);
% y = baseline1(trainX, trainY, testX, testY, istrain);
% y = baseline2(trainX, trainY, testX, testY, istrain);

[Ntest, ~] = size(testX);

if istrain == 1
    model = libsvmtrain(trainY, trainX, '-b 1');
    save('parameters_w_svm.mat', '-struct', 'model');
else
    model = load('parameters_w_svm.mat');
end
[predictlabel,accuracy,dec_values]=libsvmpredict(testY, testX, model, '-b 1');

precision = sum((predictlabel==1)&(testY(:,1)==1))/sum(predictlabel==1);
recall = sum((predictlabel==1)&(testY(:,1)==1))/sum(testY(:,1)==1);
accuracy1 = sum((predictlabel-testY(:,1)==0))/Ntest;
F1 = 2/(1/precision+1/recall);
auc = roc_curve(dec_values(:,2),testY(:,1));

fprintf('precision = %f, recall = %f, accuracy = %f, F1 = %f, auc = %f\n', precision, recall, accuracy1, F1, auc);
y = [precision, recall, accuracy1, F1, auc];

end