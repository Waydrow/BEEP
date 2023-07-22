function y = likelihood_b2_1(X,para)

[N,~] = size(X);
likelihood = 0;

for n = 1:N
    g1 = max(normpdf(max(X(n,1), 0.001),para(1),para(2)),0.001);
    g2 = max(normpdf(max(X(n,2), 0.001),para(3),para(4)),0.001);
    g3 = max(normpdf(max(X(n,3), 0.001),para(5),para(6)),0.001);
    likelihood = likelihood + log(g1) + log(g2) + log(g3);
end

y = -likelihood;

end
