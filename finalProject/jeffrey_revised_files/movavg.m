function mpo = movavg(mat, M)
    if nargin < 2
        M = 2;
    end
    
    mpo = zeros( size(mat,2) );
    
    for i=1:size(mat,2)
        mpo(max(1,i-M):min(size(mat,2),i+M),i) = 1;
        mpo(:, i) = mpo(:,i) / sum(mpo(:,i));
    end
end
    
