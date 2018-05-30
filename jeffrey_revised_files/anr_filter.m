function qo = anr_filter(in, a, b)
    qo = zeros(size(in));
    qo(:,1) = 0.9*in(:,1);
    for i=2:size(in, 2) 
        cond = in(:,i) >= qo(:,i-1);
        qo(:,i) = cond.*(a*qo(:,i-1) + (1-a)*in(:,i)) ... 
            + ~cond.*(b*qo(:,i-1)+(1-b)*in(:,i));
    end
end