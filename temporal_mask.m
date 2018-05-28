function rsp = temporal_mask(in, lambda, mu)
    tmo = zeros(size(in));
    rsp = zeros(size(in));
    tmo(:,1) = in(:,1);
    rsp(:,1) = in(:,1);
    for i=2:size(tmo, 2)
        cond = in(:,i) >= lambda*tmo(:,i-1);
        tmo(:,i) = max(lambda * tmo(:,i-1), in(:,i));
        rsp(:,i) = cond.*in(:,i) + ~cond.*(mu*tmo(:,i-1));
    end
end