module RCAM
#Residual Constrained Alternating Minimization
    
    using Distributed
    using DistributedArrays
    using LinearAlgebra
    include("common.jl")
    include("dclr.jl")
    include("dclrMN.jl")

end

