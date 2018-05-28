function so = spectral_smoothing(q, r, N)
    so = r./q;
    
    filter = movavg(so', N);
    so = filter * so;
end