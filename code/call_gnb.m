
clear, clc
data_t = load('feature_1h.txt');
data_w = load('feature_weibo_5h.txt');

data_t = data_t(:,1:6);
data_w = [data_w(:,1:4), data_w(:,7), data_w(:,9)];

% 1h: 7->30%, 8->25%, 10->20%, 12->15%, 18->10%(best)
result = data_process(data_t, 7, 0);

% 2->20%, 3->15%, 4->10%, 5->5%, 6->1%
% result = data_process(data_w, 2, 0);

