function y = likelihood_b01_1(X, X1, para)

% whether consider the dependencies of variables
global isIndependent;

[N,~] = size(X);
likelihood = 0;

for n = 1:N

if isIndependent == 1
%% If variables are mutual independent, adopt the following method
    Gamma1 = max(gampdf(max(X(n,2),0.001),para(1),para(2)),0.001);
    Gamma2 = max(gampdf(max(X(n,3),0.001),para(3),para(4)),0.001);
    Gaussian = max(normpdf(X(n,4),para(5),para(6)),0.001);
%     fprintf('n=%d : beta=%f, gamma1=%f, gamma2=%f, gaussian=%f\n', n, Beta, Gamma1, Gamma2, Gaussian);
    likelihood = likelihood + log(Gamma1) + log(Gamma2)+ log(Gaussian);
%% end of independent situation
end % end if
end % end for

y = -likelihood;

end % end function

