% GNB
function  result = baseline2(trainX,trainY,testX,testY, istrain)
time0 = cputime;

[Ntrain,~] = size(trainX);
[Ntest,~] = size(testX);
pc = [sum(trainY(:,1))/Ntrain,1-sum(trainY(:,1))/Ntrain];
po_c = zeros(Ntest,2);
pco = zeros(Ntest,2);
hpara = zeros(6,1);
npara = zeros(6,1);

if istrain == 1
options = optimoptions('fmincon','Display','off');
[hpara,~] = fmincon(@(para) likelihood_b2_1(trainX(find(trainY(:,1)==1),:),para),rand(6,1),[],[],[],[],[0,0,0,0,-200,0],[200,200,200,200,200,200], [] ,options);
[npara,~] = fmincon(@(para) likelihood_b2_1(trainX(find(trainY(:,1)==0),:),para),rand(6,1),[],[],[],[],[0,0,0,0,-200,0],[200,200,200,200,200,200], [],options);

fid = fopen('parameters.txt', 'w');
    for i = 1:6
        fprintf(fid, '%f ', hpara(i));
    end
    fprintf(fid, '\n');

    for i = 1:6
        fprintf(fid, '%f ', npara(i));
    end
    fprintf(fid, '\n');
    fclose(fid);

else
fid = fopen('parameters_t_gnb.txt', 'r');
    hpara = fscanf(fid, '%f', 6);
    npara = fscanf(fid, '%f', 6);
    fclose(fid);
end

po_c(:,1) = pc(1) .* max(normpdf(max(testX(:,1), 0.001),hpara(1),hpara(2)), 0.001) .* max(normpdf(max(testX(:,2), 0.001),hpara(3),hpara(4)), 0.001) .* max(normpdf(max(testX(:,3), 0.001),hpara(5),hpara(6)), 0.001);
po_c(:,2) = pc(2) .* max(normpdf(max(testX(:,1), 0.001),npara(1),npara(2)), 0.001) .* max(normpdf(max(testX(:,2), 0.001),npara(3),npara(4)), 0.001) .* max(normpdf(max(testX(:,3), 0.001),npara(5),npara(6)), 0.001);
pco = [po_c(:,1) ./ (po_c(:,1) + po_c(:,2)) , po_c(:,2) ./ (po_c(:,1) + po_c(:,2))];
predictlabel = (pco(:,1)>=pco(:,2));
predictcount = zeros(Ntest,1);

%for n = 1 : Ntest
%   predictcount1 = solve('normcdf(x,mu2,sigma2)/(1-normcdf(x,mu1,sigma1) + normcdf(x,mu2,sigma2))-pco(n,2)=0','x'); 
%   predictcount2 = solve('(1-normcdf(x,mu1,sigma1))/(1-normcdf(x,mu1,sigma1) + normcdf(x,mu2,sigma2))-pco(n,2)=0','x');
%   predictcount(n,1) = (predictcount1+predictcount2)/2;
%end

%Precision/Recall/F1 score
precision = sum((predictlabel==1)&(testY(:,1)==1))/sum(predictlabel==1);
recall = sum((predictlabel==1)&(testY(:,1)==1))/sum(testY(:,1)==1);
accuracy = sum((predictlabel-testY(:,1)==0))/Ntest;
F1 = 2/(1/precision+1/recall);
auc = roc_curve(pco(:,1),testY(:,1));

fprintf('precision = %f, recall = %f, accuracy = %f, F1 = %f, auc = %f\n', precision, recall, accuracy, F1, auc);
result = [precision, recall, accuracy, F1, auc];

time = cputime - time0;
fprintf('Baseline 1 finished. Time consumes %d s \n',time);
end