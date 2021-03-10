## Inter-Worker Communication Residual Constrained Alternating Minimization
# Function Definitions

function iwcpAltMinChol{ET<:Number}(b::AbstractArray{ET}, r::Int, K::Int)
# Performs RCAM on a single matricised monochromatic slice
# Calls: minFrobQR

    # Generate L0
    m::Int,n::Int = size(b)

    # May need to micro-manage this distribution
    Ld= distribute(collect(ET, randn(m,r)))
    Rd= distribute(zeros(ET,m,r))
    
    # Find non-zero inds for each col, reduce data into collections
    indn = (1:n)::UnitRange{Int64}

    # This global declaration is a workaround to open issue #15276
    const indnzd = distribute([find(b[:,i] .!= 0) for i::Int in indn])
    const bsubd = distribute([b[indnzd[i],i] for i in indn])
    
    # Find non zero inds for each row, reduce data into collections
    indm = (1:m)::UnitRange{Int64}
    const indnz_trand = distribute([find(b[i,:] .!= 0) for i::Int in indm])
    const bsub_trand = distribute([b[i,indnz_trand[i]] for i in indm])
    
    b = []

    # Begin Alterations
    for k = 1:K

        # Solve for R
        remotes1 = [@spawnat p altupdate!(Rd, Ld, bsubd, indnzd) for p in workers()]

        # Wait for all updates
        [wait(remotes1[w]) for w = 1:length(remotes1)]

        # Solve for L
        remotes2 = [@spawnat p altupdate!(Ld, Rd, bsub_trand, indnz_trand) for p in workers()]

        # Wait for all updates
        [wait(remotes2[w]) for w = 1:length(remotes2)]

    end
   
   return Ld, Rd

end
export iwcpAltMinChol

function altupdate!{ET<:Number}(Rd::AbstractArray{ET}, Ld::AbstractArray{ET}, bsubd::AbstractArray,
                                indnzd)
# Updates the local portion of R on a single worker

    #Get info about local part of darrays
    m = localindexes(Ld)

    # Get local part
    R_tmp = localpart(Rd)

    # Update the ith row of Rd
    for i::Int in m[1]
        
        # Solve for what this row or R will be updated to
        R_update = collect(ET,conj(pminFrobRsubChol(convert(Array{ET},Ld[indnzd[i],:]), bsubd[i])))
        
        # Update this row of R
        setindex!(R_tmp,vec(R_update),i - m[1][1] + 1,:)

    end

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
export pminFrobRsubChol

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
