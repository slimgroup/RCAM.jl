## Residual Constrained Alternating Minimization
# Function Definitions

function AltMinFrob{ET<:Number}(b::AbstractArray{ET}, eta::Number, r::Int, K::Int, )
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
      @time R = minFrobFrob(L,b, indi, eta)
        
        # Solve for L
      @time L = minFrobFrob(R,b',indi, eta)

    end
    
    return L, R
end
export AltMinFrob

function minFrobFrob{ET<:Number}(L::AbstractArray{ET}, b::AbstractArray{ET}, indi::UnitRange{Int64}, eta::Number)
# Solves for each row of input
# Calls: minFrobRsub

    m = size(b,2)::Int
    r = size(L,2)::Int
     
    # Initialize R
    R = zeros(ET,m,r)
    
    for k =1:length(indi)

        if indi[k]<=m

            ind = find(b[:,indi[k]])
            b_in = b[ind,indi[k]]
            A_in = L[ind, :]

            R[indi[k],:] = pminnorm(A_in, b_in, indi[k], eta)
        end

    end
    
    R = conj(R)

    return R
end

function pminnorm{ETA<:Number, ETb<:Number, TA<:AbstractArray{ETA}, Tb<:AbstractArray{ETb}}(
                                                                A::TA,
                                                                b::Tb,
                                                                k::Int,
                                                                eta::Number)
# Solves a single row, meant to be mapped over a DArray
    
    Q::TA = svdfact(A)[:U]
    tmp1 = b - Q*Q'*b
    d = eta^2 - norm(tmp1)^2

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
    M = tmp + c*eye(eltype(tmp), size(tmp)[1], size(tmp)[2]) 

    tmp2 = G'*b2
    v = M\tmp2

    end
    return v
end
export pminnorm


