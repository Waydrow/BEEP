% Seismic
clear, clc
% tweet_follower = textread('tweet_follower_5h.txt');
% tweet_time = textread('tweet_time_5h.txt');

tweet_follower = textread('weibo_follower_5h.txt');
tweet_time = textread('weibo_time_5h.txt');

[Ntotal, NCol] = size(tweet_time);
testY = zeros(Ntotal, 2);
% twitter
% 28->10%, 20->15%, 14->20%, 11->25%, 9->30%
% weibo
% 24->1%, 10->5%, 7->10%, 5->15%, 3->20%
threshold = 3;
for n = 1:Ntotal
    tweet_time(n,:) = tweet_time(n,:) - tweet_time(n,1) + 1;
    index = find(tweet_time(n,:)>=0);
    testY(n,1) = index(end);
    if testY(n,1) >= threshold
        testY(n,2) = 1;
    end
end
tweet_time = tweet_time/60;

[hot, ~] = size(find(testY(:,2)==1));
fprintf('hot/all = %f\n', hot/Ntotal);

%{
% note: generate follower data for weibo dataset
[Ntotalw, NColw] = size(weibo_time);
weibo_follower = zeros(Ntotalw, NColw);
temp_follower = [];
for n = 1:Ntotal
    index = find(tweet_time(n,:)>=0);
    a = index(end);
    temp_follower = [temp_follower, tweet_follower(n,index)];
end

i = 1;
fid = fopen('weibo_follower_25h.txt', 'w');
for n = 1:Ntotalw
   index = find(weibo_time(n,:)>0);
   a = index(end);
   for j = 1:a
       weibo_follower(n,j) = temp_follower(i);
       fprintf(fid, '%d ', temp_follower(i));
       i = i + 1;
   end
   fprintf(fid, '\n');
end
fclose(fid);
%}

result = baseline4(tweet_time, tweet_follower, testY, threshold);