## Residual Constrained Alternating Minimization
# Function Definitions

function AltMinBackslash(b,r,K)
# Performs RCAM on a single matricised monochromatic slice
# Calls: minFrob

    # Generate L0
    m,n = size(b)
    L = randn(m,r)
   
    local R

    # Begin Alterations
    for k = 1:K

        # Solve for R
        R = minFrob(L,b)
        
        # Solve for L
        L = minFrob(R,b')

    end
    
    return L, R
end
export AltMinBackslash


function minFrob(L,b)
# Solves for each row of input
# Calls: minFrobRsub

    m = size(b,2)
    r = size(L,2)
     
    # Initialize R
    R = zeros(m,r)
    if ~isreal(b)
        R = complex(R)
    end


    # Solve for each row of R
    for k=1:m

        R[k,:] = minFrobRsub(L,b,k)

    end
    
    R = conj(R)

    return R
end


function minFrobRsub(L,b,k)
# Solves a single row

    # Solve for the kth row of R
    bsub = b[:,k]
    ind = find(x -> x != 0, bsub)
    bsub = bsub[ind]
    L = L[ind,:]

    v = L\bsub
    
    return v

end

