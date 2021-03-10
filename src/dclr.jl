export dclr

function dclr(b::AbstractArray{ET}, r::Int, K::Int) where ET<:Number
# Performs RCAM on a single matricised monochromatic slice
# Calls: minFrobQR
    
    # Get worker IDs 
    wids = sort(workers())
    nworkers = length(wids)
   
    if nworkers < 3
        error(string("DCLR method requires atleast three workers, got ", nworkers))
    end

    # Dedicate two workers to distribute/store L and R
    ownerL = wids[1]
    ownerR = wids[2]
    
    # Dedicate remaining workers to as solvers
    solvers = wids[3:nworkers]
    nsolvers = length(solvers)
    
    # Give this info to every solver
    [sendto(pid, ownerL = ownerL, ownerR = ownerR) for pid in procs()] #change to procs()[solvers] 
    
    # Get size of data inorder to generate L0
    m::Int,n::Int = size(b)

    #Create L and R on their owners
    sendto(ownerL, L=collect(ET, randn(m,r)))
    sendto(ownerR, R=zeros(ET,n,r))

    # Find non-zero inds for each col, reduce data into collections
    indn = (1:n)::UnitRange{Int64}

    # This global declaration is a workaround to open issue #15276
    indnzd = distribute([findall(!iszero,b[:,i]) for i::Int in indn],
                        procs = procs()[solvers])
    bsubd = distribute([b[indnzd[i],i] for i in indn],
                        procs = procs()[solvers])
    
    # Find non zero inds for each row, reduce data into collections
    indm = (1:m)::UnitRange{Int64}
    indnz_trand = distribute([findall(!iszero,b[i,:]) for i::Int in indm],
                        procs = procs()[solvers])
    bsub_trand = distribute([b[i,indnz_trand[i]] for i in indm],
                        procs = procs()[solvers])
    
    # Clear data
    b = []

    # Begin Alterations
    for k = 1:K

        @time @sync for pid in procs()[solvers]
            @async fetch(@spawnat pid altupdateR!(ownerL, ownerR, bsubd, indnzd))
        end

        @time @sync for pid in procs()[solvers]
            @async fetch(@spawnat pid altupdateL!(ownerL, ownerR, bsub_trand,indnz_trand))
        end

    end
   
   return ownerL, ownerR 

end

function altupdateL!(ownerL::Int, ownerR::Int, bsubd::AbstractArray{AT}, indnzd) where AT
    
    ET = eltype(AT)
    
    #Get info about local part of data
    m = localindices(bsubd)[1]

    # Get updated version of R
    R_tmp = fetch(@spawnat(ownerR, Main.R)) 
    
    # Update the ith row of Rd
    for i in m
        
        local_i = i - m[1] + 1

        # Ask for required part of L
        req_rows = DistributedArrays.localpart(indnzd)[local_i]

        # Solve for what this row or R will be updated to
        L_update = collect(ET,conj(pminFrobRsubChol(R_tmp[req_rows,:],
                            conj(DistributedArrays.localpart(bsubd)[local_i]))))
        
        # Create a future for this result
        L_future = Future(ownerL)
        put!(L_future, L_update)

        # Update the i-th row of R with the future 
        @spawnat(ownerL, setindex!(Main.L, fetch(L_future), i, :))

    end

end


function altupdateR!(ownerL::Int, ownerR::Int, bsubd::AbstractArray{AT}, indnzd) where AT
    
    ET = eltype(AT)
    
    #Get info about local part of data
    m = localindices(bsubd)[1]

    # Get updated version of L
    L_tmp = fetch(@spawnat(ownerL, Main.L)) 
    
    # Update the ith row of Rd
    for i in m
        
        local_i = i - m[1] + 1

        # Ask for required part of L
        req_rows = DistributedArrays.localpart(indnzd)[local_i]

        # Solve for what this row or R will be updated to
        R_update = collect(ET,conj(pminFrobRsubChol(L_tmp[req_rows,:],
                            DistributedArrays.localpart(bsubd)[local_i])))
        
        # Create a future for this result
        R_future = Future(ownerR)
        put!(R_future, R_update)

        # Update the i-th row of R with the future 
        @spawnat(ownerR, setindex!(Main.R, fetch(R_future), i, :))

    end

end

function pminFrobRsubChol(L2::AbstractArray{ET},bsub::AbstractArray{ET}) where ET<:Number
# Solves a single row, meant to be mapped over a DArray

    bsub = L2'*bsub 
    L2 =L2'*L2
    
    #fL2 = chol(L2)
    #v = backsub(fL2,fwdsub((fL2)',bsub)) 
    v = L2\bsub

    return v
end
export pminFrobRsubChol

function backsub(A::AbstractArray{ET}, y::AbstractArray{ET}) where ET<:Number

    m = size(A,2)
    x = zeros(ET,m,1)
    x[m] = y[m]/A[m,m]
    for i = m-1:-1:1
        x[i] = (y[i]-dot(A[i,i+1:m], x[i+1:m]))/A[i,i]
    end

    return x
end

function fwdsub(A::AbstractArray{ET},y::AbstractArray{ET}) where ET<:Number

    m = size(A,2)
    x = zeros(ET,m,1)
    x[1] = y[1]/A[1,1]
    for i = 2:1:m
        x[i] = (y[i]-dot(A[i,i-1:-1:1], x[i-1:-1:1]))/A[i,i]
    end

    return x
end
