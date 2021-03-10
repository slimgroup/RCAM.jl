## Residual Constrained Alternating Minimization
# Function Definitions

function pAltMinChol{ET<:Number}(b::AbstractArray{ET}, r::Int, K::Int)
# Performs RCAM on a single matricised monochromatic slice
# Calls: minFrobQR

    # Generate L0
    m,n = size(b)
    L = collect(ET, randn(m,r))

    indn = (1:n)::UnitRange{Int64}
    local R
   
    # Reduce the data down to nonzero elements of each column
    ### change to do in one loop ###
    indnz = [find(x -> x != 0, b[:,i]) for i in indn]
    bsub = [b[indnz[i],i] for i in indn]
    
    # This needs to be cleaned up for non-square entries
    indnz_tran = [find(x -> x != 0, b[i,:]) for i in indn]
    bsub_tran = [b[i,indnz_tran[i]] for i in indn]

    # Distribute the reduced data
    bsubD = distribute(bsub)
    bsubD_tran = distribute(bsub_tran)

    # Begin Alterations
    for k = 1:K

        # Solve for R
        R = pminFrobChol(L,bsubD,indn,indnz)
        
        # Solve for L
        L = pminFrobChol(R,bsubD_tran,indn,indnz_tran)

    end
    
    return L, R
end
export pAltMinChol

function pminFrobChol{ET<:Number}(L::AbstractArray{ET}, bsubD::AbstractArray, indi::UnitRange{Int64},
                    indnz::AbstractArray)
# Solves for each row of input
# Calls: minFrobRsub
    
    # Init L2 and bsub
    L2 = [L[indnz[i],:] for i in indi]

    # Distribute L2 and bsub
    LD = distribute(L2)

    # Map solving function to distributed inputs
    Rcol = map(pminFrobRsubChol, LD, bsubD)

    # Reform R to matrix
    R = Array{ET}(size(L))
    for i = 1:length(Rcol)
        R[i,:] = Rcol[i]
    end
    
    R = conj(R)

    return R
end

function pminFrobRsubChol{ET<:Number}(L2::AbstractArray{ET},bsub::AbstractArray{ET})
# Solves a single row, meant to be mapped over a DArray

    bsub = transpose(L2)*bsub 
        
    # Chol fact
    L2 = L2'*L2

    # Faster to do in w/o mem alloc, even though it does chol twice
    v = backsub(chol(L2),fwdsub((chol(L2))',bsub)) 
    
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
