function y = likelihood_b0_1(X, X1, para)

% whether consider the dependencies of variables
global isIndependent;

[N,~] = size(X);
likelihood = 0;
% following para(1), para(2), para(3), para(4) are parameters of dirichlet distribution
beta = max(gamma(para(1)) * gamma(para(2)) * gamma(para(3)) * gamma(para(4)) / gamma(para(1) + para(2) + para(3) + para(4)), 0.001);

for n = 1:N

if isIndependent == 1
%% If variables are mutual independent, adopt the following method
    Beta = max(betapdf(max(X(n,1),0.001),para(1),para(2)),0.001);
    Gamma1 = max(gampdf(max(X(n,2),0.001),para(3),para(4)),0.001);
    Gamma2 = max(gampdf(max(X(n,3),0.001),para(5),para(6)),0.001);
    Gaussian = max(normpdf(X(n,4),para(7),para(8)),0.001);
%     fprintf('n=%d : beta=%f, gamma1=%f, gamma2=%f, gaussian=%f\n', n, Beta, Gamma1, Gamma2, Gaussian);
    likelihood = likelihood + log(Beta) + log(Gamma1) + log(Gamma2)+ log(Gaussian);
%% end of independent situation

else

%% If consider the dependencies of the variables, following method is better
%     To sovle the dependencies problem, make transformation of some variables as follows. 
% 
%     S' = S/(2*max(Si)), V1' = (T*V1) / (4*max(Si)), V2' = (T*V2) / (4*max(Si))
%     (S', V1', V2') ~ dirichlet(para(1), para(2), para(3), para(4))
%     (V1, V2, A) ~ multi gaussian(para(5), para(6); para(7), para(8), para(9), para(10)) 
%     V1 ~ gamma(para(11), para(12))
%     V2 ~ gamma(para(13), para(14))

%     fprintf('S+V1+V2=%f\n', X1(n,1)+X1(n,2)+X1(n,3));
    dirichlet = min(X1(n,1)^(para(1)-1) * X1(n,2)^(para(2)-1) * X1(n,3)^(para(3)-1) * (1-X1(n,1)-X1(n,2)-X1(n,3))^(para(4)-1) / beta, 500);
    dirichlet = max(dirichlet, 0.001);
    gaussian = max(mvnpdf([X(n,2), X(n,3), X(n,4)], [para(5) para(6) para(7)], [para(8) 0 0; 0 para(9) 0; 0 0 para(10)]), 0.001);
    gamma1 = max(gampdf(max(X(n,2),0.001), para(11), para(12)), 0.001);
    gamma2 = max(gampdf(max(X(n,3),0.001), para(13), para(14)), 0.001);
%     fprintf('n=%d -> betafunc=%f, dirichlet=%f, gaussian=%f, gamma1=%f, gamma2=%f\n', n, beta, dirichlet, gaussian, gamma1, gamma2);

    likelihood = likelihood + log(dirichlet) + log(gaussian) - log(gamma1) - log(gamma2);
%% end of dependent situation
end % end if
end % end for

y = -likelihood;

end % end function

