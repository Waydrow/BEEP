% 注：由于SVM训练非常快，所以每次实验无需保存参数
clear, clc
data_t = load('feature_1h.txt');
data_w = load('feature_weibo_5h.txt');

data_t = [data_t(:,1), data_t(:,17), data_t(:,16)];
data_w = [data_w(:,1), data_w(:,23), data_w(:,22)];

% twitter
% 1h: 7->30%, 8->25%, 10->20%, 12->15%, 18->10%(best)
% result = data_process_svm(data_t, 7, 1);

% weibo
% 2->20%, 3->15%, 4->10%, 5->5%, 6->1%
result = data_process_svm(data_w, 2, 1);

