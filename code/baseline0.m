% BEEP
function  result = baseline0(trainX, trainX1, trainY,testX, testX1, testY)
% trainX为训练集，行数为样本数，每一列为一个feature
% trainY为0,1序列，1:hot event, 0:normal event
% test 为测试集

global isTraining;
global isIndependent;
time0 = cputime;

[Ntrain,~] = size(trainX);
[Ntest,~] = size(testX);
pc = [sum(trainY(:,1))/Ntrain,1-sum(trainY(:,1))/Ntrain];
po_c = zeros(Ntest,2);
pco = zeros(Ntest,2);



%% Train Model
numForLike1 = 8;
% numForLike1 = 14;
hpara = zeros(numForLike1,1);
npara = zeros(numForLike1,1);
hpara1 = zeros(4,1);
npara1 = zeros(4,1);
if isTraining == 1
%*************** Estimate Parameters *************************%

    % options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    % options = optimoptions('fmincon','Display','off','OptimalityTolerance', 0);
    options = optimoptions('fmincon', 'Display', 'off');
    % [hpara,fval] = fmincon(@(para) likelihood_b2_1(trainX(find(trainY(:,1)==1),:),para),[rand(7,1);1;0;0;0;1;0;0;0;1;rand(4,1)],[],[],[],[],[zeros(4,1);repmat(-200,3,1);zeros(13,1)],repmat(200,20,1));
    % [npara,fval] = fmincon(@(para) likelihood_b2_1(trainX(find(trainY(:,1)==0),:),para),[rand(7,1);1;0;0;0;1;0;0;0;1;rand(4,1)],[],[],[],[],[zeros(4,1);repmat(-200,3,1);zeros(13,1)],repmat(200,20,1));
    
    if isIndependent == 1
        % independent situation
        [hpara,fval] = fmincon(@(para) likelihood_b0_1(trainX(find(trainY(:,1)==1),:), [], para),rand(8,1),[],[],[],[],[zeros(6,1);-200;0],repmat(200,8,1), [] ,options);
        [npara,fval] = fmincon(@(para) likelihood_b0_1(trainX(find(trainY(:,1)==0),:), [], para),rand(8,1),[],[],[],[],[zeros(6,1);-200;0],repmat(200,8,1), [] ,options);
        [hpara1,fval] = fmincon(@(para) likelihood_b0_2(trainX(find(trainY(:,1)==1),:),para),rand(4,1),[],[],[],[],[0,0,0,0],[200,200,200,200], [] ,options);
        [npara1,fval] = fmincon(@(para) likelihood_b0_2(trainX(find(trainY(:,1)==0),:),para),rand(4,1),[],[],[],[],[0,0,0,0],[200,200,200,200], [] ,options);
        
    else
        % dependent situation
        [hpara, ~] = fmincon(@(para) likelihood_b0_1(trainX(find(trainY(:,1)==1),:), trainX1(find(trainY(:,1)==1),:), para),rand(14,1),[],[],[],[],[zeros(4,1);repmat(-200,3,1);zeros(7,1)],repmat(200,14,1));
        fprintf('likelihood 1 of hot events is finished.\n');
        [npara, ~] = fmincon(@(para) likelihood_b0_1(trainX(find(trainY(:,1)==0),:), trainX1(find(trainY(:,1)==0),:), para),rand(14,1),[],[],[],[],[zeros(4,1);repmat(-200,3,1);zeros(7,1)],repmat(200,14,1));
        fprintf('likelihood 1 of normal events is finished.\n');
        [hpara1,~] = fmincon(@(para) likelihood_b0_2(trainX(find(trainY(:,1)==1),:),para),rand(4,1),[],[],[],[],[0,0,0,0],[200,200,200,200], [] ,options);
        fprintf('likelihood 2 of hot events is finished.\n');
        [npara1,~] = fmincon(@(para) likelihood_b0_2(trainX(find(trainY(:,1)==0),:),para),rand(4,1),[],[],[],[],[0,0,0,0],[200,200,200,200], [] ,options);
        fprintf('likelihood 2 of normal events is finished.\n');
    end % end if
%*************** Save Estimated Parameters in File ***************%
    fid = fopen('parameters.txt', 'w');
    for i = 1:numForLike1
        fprintf(fid, '%f ', hpara(i));
    end
    fprintf(fid, '\n');

    for i = 1:numForLike1
        fprintf(fid, '%f ', npara(i));
    end
    fprintf(fid, '\n');

    for i = 1:4
        fprintf(fid, '%f ', hpara1(i));
    end
    fprintf(fid, '\n');

    for i = 1:4
        fprintf(fid, '%f ', npara1(i));
    end
    fprintf(fid, '\n');

    fclose(fid);

%%
else % isTraining == 0
%% Model has been trained
% Read Parameters from File
    fid = fopen('parameters_t_i_beep.txt', 'r');
    hpara = fscanf(fid, '%f', numForLike1);
    npara = fscanf(fid, '%f', numForLike1);
    hpara1 = fscanf(fid, '%f', 4);
    npara1 = fscanf(fid, '%f', 4);
    fclose(fid);
end
%%


%% Test data
if isIndependent == 1
    % independent situation
    %beta = max(gamma(hpara(1))*gamma(hpara(2))*gamma(hpara(3))*gamma(hpara(4))*gamma(hpara(5))/gamma(hpara(1)+hpara(2)+hpara(3)+hpara(4)+hpara(5)),0.001);
    %Dirichlet = testX1(:,1).^hpara(1) .* testX1(:,2).^hpara(2) .* testX1(:,3).^hpara(3) .* testX1(:,4).^hpara(4) .* (1-testX1(:,1)-testX1(:,2)-testX1(:,3)-testX1(:,4)).^hpara(5) ./ beta;
    %Gaussian = mvnpdf([testX(:,2),testX(:,3),testX(:,4)],[hpara(5) hpara(6) hpara(7)],[hpara(8) 0 0;0 hpara(12) 0;0 0 hpara(16)]);
    %Gamma1 = gampdf(testX(:,2),hpara(17),hpara(18));
    %Gamma2 = gampdf(testX(:,3),hpara(19),hpara(20));
    Beta = max(betapdf(max(testX(:,1),0.001),hpara(1),hpara(2)),0.001);
    Gamma1 = max(gampdf(max(testX(:,2),0.001),hpara(3),hpara(4)),0.001);
    Gamma2 = max(gampdf(max(testX(:,3),0.001),hpara(5),hpara(6)),0.001);
    Gaussian = max(normpdf(testX(:,4),hpara(7),hpara(8)),0.001);
    Gamma3 = max(gampdf(max(testX(:,5),0.001),hpara1(1),hpara1(2)),0.001);
    Gamma4 = max(gampdf(max(testX(:,6),0.001),hpara1(3),hpara1(4)),0.001);
    po_c(:,1) = pc(1) .* Beta .* Gamma1 .* Gamma2 .* Gaussian .* Gamma3 .* Gamma4;
    %beta = max(gamma(npara(1))*gamma(npara(2))*gamma(npara(3))*gamma(npara(4))*gamma(npara(5))/gamma(npara(1)+npara(2)+npara(3)+npara(4)+npara(5)),0.001);
    %Dirichlet = testX1(:,1).^npara(1) .* testX1(:,2).^npara(2) .* testX1(:,3).^npara(3) .* testX1(:,4).^npara(4) .* (1-testX1(:,1)-testX1(:,2)-testX1(:,3)-testX1(:,4)).^npara(5) ./ beta;
    %Gaussian = mvnpdf([testX(:,2),testX(:,3),testX(:,4)],[hpara(5) hpara(6) hpara(7)],[hpara(8) 0 0;0 hpara(12) 0;0 0 hpara(16)]);
    %Gamma1 = gampdf(testX(:,2),hpara(17),hpara(18));
    %Gamma2 = gampdf(testX(:,3),hpara(19),hpara(20));
    Beta = max(betapdf(max(testX(:,1),0.001),npara(1),npara(2)),0.001);
    Gamma1 = max(gampdf(max(testX(:,2),0.001),npara(3),npara(4)),0.001);
    Gamma2 = max(gampdf(max(testX(:,3),0.001),npara(5),npara(6)),0.001);
    Gaussian = max(normpdf(testX(:,4),npara(7),npara(8)),0.001);
    Gamma3 = max(gampdf(max(testX(:,5),0.001),npara1(1),npara1(2)),0.001);
    Gamma4 = max(gampdf(max(testX(:,6),0.001),npara1(3),npara1(4)),0.001);
    po_c(:,2) = pc(2) .* Beta .* Gamma1 .* Gamma2 .* Gaussian .* Gamma3 .* Gamma4;

else % isIndependent == 0
    % dependent situation
    beta = max(gamma(hpara(1)) * gamma(hpara(2)) * gamma(hpara(3)) * hpara(4) / gamma(hpara(1) + hpara(2) + hpara(3) + hpara(4)), 0.001);
    dirichlet = testX1(:,1).^(hpara(1)-1) .* testX1(:,2).^(hpara(2)-1) .* testX1(:,3).^(hpara(3)-1) .* (1-testX1(:,1)-testX1(:,2)-testX1(:,3)).^hpara(4) ./ beta;
    gaussian = mvnpdf([testX(:,2),testX(:,3),testX(:,4)], [hpara(5), hpara(6), hpara(7)], [hpara(8) 0 0; 0 hpara(9) 0; 0 0 hpara(10)]);
    gamma1 = max(gampdf(max(testX(:,2),0.001), hpara(11), hpara(12)), 0.001);
    gamma2 = max(gampdf(max(testX(:,3),0.001), hpara(13), hpara(14)), 0.001);
    gamma3 = max(gampdf(max(testX(:,5),0.001), hpara1(1), hpara1(2)), 0.001);
    gamma4 = max(gampdf(max(testX(:,5),0.001), hpara1(3), hpara1(4)), 0.001);
    po_c(:,1) = pc(1) .* dirichlet .* gaussian ./ (gamma1 .* gamma2) .* gamma3 .*gamma4;

    beta = max(gamma(npara(1)) * gamma(npara(2)) * gamma(npara(3)) * npara(4) / gamma(npara(1) + npara(2) + npara(3) + npara(4)), 0.001);
    dirichlet = testX1(:,1).^(npara(1)-1) .* testX1(:,2).^(npara(2)-1) .* testX1(:,3).^(npara(3)-1) .* (1-testX1(:,1)-testX1(:,2)-testX1(:,3)).^npara(4) ./ beta;
    gaussian = mvnpdf([testX(:,2),testX(:,3),testX(:,4)], [npara(5), npara(6), npara(7)], [npara(8) 0 0; 0 npara(9) 0; 0 0 npara(10)]);
    gamma1 = max(gampdf(max(testX(:,2),0.001), npara(11), npara(12)), 0.001);
    gamma2 = max(gampdf(max(testX(:,3),0.001), npara(13), npara(14)), 0.001);
    gamma3 = max(gampdf(max(testX(:,5),0.001), npara1(1), npara1(2)), 0.001);
    gamma4 = max(gampdf(max(testX(:,5),0.001), npara1(3), npara1(4)), 0.001);
    po_c(:,2) = pc(2) .* dirichlet .* gaussian ./ (gamma1 .* gamma2) .* gamma3 .*gamma4;
end % end if

pco = [po_c(:,1) ./ (po_c(:,1) + po_c(:,2)) , po_c(:,2) ./ (po_c(:,1) + po_c(:,2))];
predictlabel = (pco(:,1)>=pco(:,2));
%%

fprintf('predict hot = %d, right = %d; normal = %d, right = %d\n', sum(predictlabel==1), sum((predictlabel==1)&(testY(:,1)==1)), sum(predictlabel==0), sum((predictlabel==0)&(testY(:,1)==0)));
fprintf('prediction right / all = %d / %d\n', sum((predictlabel-testY(:,1)) == 0), size(testY,1));
%% metrics
%Precision/Recall/F1 score
precision = sum((predictlabel==1)&(testY(:,1)==1))/sum(predictlabel==1);
recall = sum((predictlabel==1)&(testY(:,1)==1))/sum(testY(:,1)==1);
accuracy = sum((predictlabel-testY(:,1)==0))/Ntest;
F1 = 2/(1/precision+1/recall);
auc = roc_curve(pco(:,1), testY(:,1));
%%

fprintf('precision = %f, recall = %f, accuracy = %f, F1 = %f, auc = %f\n', precision, recall, accuracy, F1, auc);
result = [precision, recall, accuracy, F1, auc];

time = cputime - time0;
fprintf('Computation finished. Time consumes %f minutes.\n',time/60);

end
%}