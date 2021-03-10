## RCAM using dedicated owners to store L and R factors with min-norm
# Function Definitions

export dclrMN

function dclrMN(b::AbstractArray{ET}, eta::Number, r::Int, K::Int) where ET<:Number
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
            @async fetch(@spawnat pid altupdateR!(ownerL, ownerR, bsubd, indnzd, eta))
        end

        @time @sync for pid in procs()[solvers]
            @async fetch(@spawnat pid altupdateL!(ownerL, ownerR, bsub_trand,indnz_trand, eta))
        end

    end
   
   return ownerL, ownerR 

end

function altupdateL!(ownerL::Int, ownerR::Int, bsubd::AbstractArray{AT}, indnzd, eta::Number) where AT
    
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
        L_update = collect(ET,conj(pminnorm(R_tmp[req_rows,:],
                            DistributedArrays.localpart(bsubd)[local_i], eta)))
        
        # Create a future for this result
        L_future = Future(ownerL)
        put!(L_future, L_update)

        # Update the i-th row of R with the future 
        @spawnat(ownerL, setindex!(Main.L, fetch(L_future), i, :))

    end

end


function altupdateR!(ownerL::Int, ownerR::Int, bsubd::AbstractArray{AT}, indnzd, eta::Number) where AT
    
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
        R_update = collect(ET,conj(pminnorm(L_tmp[req_rows,:],
                            DistributedArrays.localpart(bsubd)[local_i], eta)))
        
        # Create a future for this result
        R_future = Future(ownerR)
        put!(R_future, R_update)

        # Update the i-th row of R with the future 
        @spawnat(ownerR, setindex!(Main.R, fetch(R_future), i, :))

    end

end

function pminnorm( A::TA, b::Tb, eta::Number) where {ETA<:Number, ETb<:Number, TA<:AbstractArray{ETA}, Tb<:AbstractArray{ETb}}
# Solves a single row, meant to be mapped over a DArray
    
    
    Q::TA = svd(A).U
    d = eta^2 - norm(b - Q*Q'*b)^2

    # Force type of v
    v::Array{ETA,1} = Array{ETA,1}()
    if d < 0
        v = A\b
    else
    
    G = Q'*A
    H = G*G'
    b2 = Q'*b
    v1 = norm(H\b2)
    c = sqrt(d)/v1
    tmp = G'*G
    M = tmp + c*Matrix{eltype(tmp)}(I, size(tmp)[1], size(tmp)[2])

    tmp2 = G'*b2
    v = M\tmp2

    end
    return v
end
export pminnorm
