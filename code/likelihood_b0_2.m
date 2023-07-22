function y = likelihood_b0_2(X,para)

[N,~] = size(X);
likelihood = 0;

for n = 1:N
    % T ~ Gamma(para(1), para(2))
    % C ~ Gamma(para(3), para(4))
    Gamma3 = max(gampdf(max(X(n,5),0.001),para(1),para(2)),0.001);
    Gamma4 = max(gampdf(max(X(n,6),0.001),para(3),para(4)),0.001);
    likelihood = likelihood + log(Gamma3) + log(Gamma4);
end

y = -likelihood;

end