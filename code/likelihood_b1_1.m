function y = likelihood_b1_1(X,para)

[N,~] = size(X);
likelihood = 0;

for n = 1:N
    Beta = max(betapdf(max(X(n,1), 0.001),para(1),para(2)),0.001);
    Gamma = max(gampdf(max(X(n,2), 0.001),para(3),para(4)),0.001);
    Gaussian = max(normpdf(max(X(n,3), 0.001),para(5),para(6)),0.001);
    likelihood = likelihood + log(Beta) + log(Gamma) + log(Gaussian);
end

y = -likelihood;

end
