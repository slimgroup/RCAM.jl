## Residual Constrained Alternating Minimization
# Function Definitions

function AltMinChol{ET<:Number}(b::AbstractArray{ET}, r::Int, K::Int)
# Performs RCAM on a single matricised monochromatic slice
# Calls: minFrobQR

    # Generate L0
    m,n = size(b)
    L = collect(ET, randn(m,r))

    indi = (1:max(m,n))::UnitRange{Int64}
    local R

    # Begin Alterations
    for k = 1:K

        # Solve for R
     @time R = minFrobChol(L,b,indi)
        
        # Solve for L
     @time L = minFrobChol(R,b',indi)

    end
    
    return L, R
end
export AltMinChol

function minFrobChol{ET<:Number}(L::AbstractArray{ET}, b::AbstractArray{ET}, indi::UnitRange{Int64})
# Solves for each row of input
# Calls: minFrobRsub

    m = size(b,2)::Int
    r = size(L,2)::Int
     
    # Initialize R
    R = zeros(ET,m,r)
    
    for k =1:length(indi)

        if indi[k]<=m

             R[indi[k],:] = minFrobRsubChol(L, b, indi[k])

        end

    end
    
    R = conj(R)

    return R
end

function minFrobRsubChol{ET<:Number}(L::AbstractArray{ET},b::AbstractArray{ET},indi::Int)
# Solves a single row

    # Solve for the kth row of R
    bsub = b[:,indi]
        
    # Find the non zero rows of the kth column of b
    ind = find(x -> x != 0, b[:,indi])
        
    # Keep only the non zero
    L2 = L[ind,:]
    bsub = transpose(L2)*bsub[ind] 
        
    # Chol fact
    L2 = L2'*L2
    Q = chol(L2)

    bsub2 = fwdsub(Q',bsub)

    v = backsub(Q,bsub2)
    
    return v

end

function backsub{ET<:Number}(A::AbstractArray{ET}, y::AbstractArray{ET})

    m = size(A,2)
    x = zeros(ET,m,1)

    x[m] = y[m]/A[m,m]

    for i = m-1:-1:1
        x[i] = (y[i]-dot(A[i,i+1:m], x[i+1:m]))/A[i,i]
    end
    
    return x
end

function fwdsub{ET<:Number}(A::AbstractArray{ET},y::AbstractArray{ET})

    m = size(A,2)
    x = zeros(ET,m,1)
    x[1] = y[1]/A[1,1]
    
    for i = 2:1:m
        x[i] = (y[i]-dot(A[i,i-1:-1:1], x[i-1:-1:1]))/A[i,i]
    end

    return x
end
