% Seismic
function result = baseline4(testT,testN,testY,threshold)
%testN = follower;
%testT = time;

time0 = cputime;
%testT每一行表示每个topic相关tweet出现   相对时间,以min为单位
%testN每一行表示每个topic相关tweet的用户拥有的follower数量

t = 60; %预测时间,以min为单位, 60, 120, 180, 240, 300
alpha = [0.389 0.803 0.772 0.709 0.68 0.562 0.454 0.378 0.352 0.326];
[Ntest,~] = size(testN);
predict_count = zeros(Ntest,1);
syms s;

for n = 1:Ntest
    Tn = testT(n,find(testT(n,:)<=t&testT(n,:)>0));
    Nn = testN(n,find(testT(n,:)<=t&testT(n,:)>0));
    [~,Rt] = size(Tn);
    Nt = sum(Nn);
    %Nte = sum( testN(find(testT(n,:)<=t)) .* int(max(0,1-2*(t-s)/t)*min(0.000627,0.000627*((s-testT(find(testT(n,:)<=t))/5)^(-1.24)),s,testT(find(testT(n,:)<=t),t))) );
    Nte = 0;
    for rt = 1:Rt
        s = testT(n,rt)+0.0001:0.01:t;
        integralpart = sum(memory_kernel(s-testT(n,rt)) * 0.01);
        Nte = Nte + testN(n,rt) * integralpart; 
    end
    pt = infectiousness(Tn,Nn,t);
    predict_count(n,1) = Rt + alpha(1,6)*pt*(Nt-Nte)/(1-2*pt);
    % fprintf('%d finishes \n',n);
end
predictlabel = (predict_count>=threshold);

%Precision/Recall/F1 score
precision = sum((predictlabel==1)&(testY(:,2)==1))/sum(predictlabel==1);
recall = sum((predictlabel==1)&(testY(:,2)==1))/sum(testY(:,2)==1);
accuracy = sum((predictlabel-testY(:,2)==0))/Ntest;
F1 = 2/(1/precision+1/recall);
auc = roc_curve(predict_count(:,1), testY(:,2));

fprintf('precision = %f, recall = %f, accuracy = %f, F1 = %f, auc = %f\n', precision, recall, accuracy, F1, auc);
result = [precision, recall, accuracy, F1, auc];

time = cputime - time0;
fprintf('Baseline 4 finished. Time consumes %d s \n',time);
end