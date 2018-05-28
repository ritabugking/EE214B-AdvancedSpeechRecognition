function mu = mean_power(in, lambda)
    mu = zeros(size(in, 2),1);
    means = mean(in);
    mu(1) = mean(in(:,1));
    for i=2:length(mu)
        mu(i) = lambda*mu(i-1) + (1-lambda)*means(i);
    end
end