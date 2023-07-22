function y = kernel_t(s,t)

if 1-2*s/t>0
    y = 1-2*s/t;
else
    y = 0;
end

end