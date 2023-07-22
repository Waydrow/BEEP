function y = memory_kernel(s)

[~,N] = size(s);
y = zeros(1,N);
for i=1:N
    if (s(i)<5)
        y(i) = 0.000627;
    else
        y(i) = 0.000627*(s(i)/5)^(-1.242);
    end
end

end