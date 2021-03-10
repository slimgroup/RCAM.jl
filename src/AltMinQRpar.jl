## Parallel Residual Constrained Alternating Minimization
# Function Definitions

function AltMinQR{ET<:Number}(b::AbstractArray{ET}, r::Int, K::Int)
# REAL
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
        R = minFrobQR(L,b,indi)
        
        # Solve for L
        L = minFrobQR(R,b',indi)

    end
    
    return L, R
end
export AltMinQR

function minFrobQR{ET<:Number}(L::AbstractArray{ET}, b::AbstractArray{ET}, indi::UnitRange{Int64})
# Solves for each row of input
# Calls: minFrobRsub

    m = size(b,2)::Int
    r = size(L,2)::Int
     
    # Initialize R
    R = zeros(ET,m,r)
    
    for k =1:length(indi)

        if indi[k]<=m

            # Solve for the kth row of R
            bsub = b[:,indi[k]]
            
            # Find the non zero rows of the kth column of b
            ind = find(x -> x != 0, b[:,indi[k]])
            
            # Keep only the non zero
            bsub = b[ind,indi[k]] 
            
            # QRfact of ind rows of L
            q_qr,r_qr = qr(L[ind,:],thin=false)
        
            #Extend r to full size
            extension = zeros(size(q_qr,1)-size(r_qr,1),size(r_qr,2))
            r_full = vcat(r_qr,extension)

            R[indi[k],:] = minFrobRsub(q_qr,r_full,bsub)

        end

    end
    
    R = conj(R)

    return R
end

function minFrobRsub(q::AbstractArray,r_full::AbstractArray,bsub::AbstractArray)
# Solves a single row

    bsub = q'*bsub

    v = backsub(r_full,bsub)
    
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
