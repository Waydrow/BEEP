function pt = infectiousness(T,N,t)

Rt = sum( max(0,1-2*(t-T)/t) );
%syms s;
Nt = 0;
[~,Tsize] = size(T);
for i = 1:Tsize
    s = T(i):0.01:t;
%     s = T(i)+0.000001:0.01:t-0.0000001;
    Nt = Nt + sum(max(N(i),1) * (kernel_t(t-s,t) .* memory_kernel(s-T(i))) * 0.01);
end
pt = Rt/max(Nt,0.001);
end