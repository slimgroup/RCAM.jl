using Printf
using Random
using LinearAlgebra
using Distributed
using MAT
@everywhere using DistributedArrays
@everywhere using RCAM
@everywhere using Random
@everywhere Random.seed!(123)

#Load the data
fid = matopen("../data/Data.mat")
d = read(fid)
X = d["D"]

function pCholtest(X)
    # Choltest is just a wrapper for this script

    # Params
    r = 50 #Rank
    alt = 5 #Iterations
    perc = 0.8 #Percent of missing values
    s1,s2 = size(X)

    # Subsample
    ind = randperm(s1*s2)
    inds = ind[1:Int(round(perc*s1*s2))]
    b = copy(X)
    b[inds] .= 0

    # Interpolate
    @time ownerL, ownerR = dclr(b, r, alt)

    # Gather L and R results
    L = fetch(@spawnat ownerL Main.L)
    R = fetch(@spawnat ownerR Main.R)

    return L,R 
end

snr(raw,interp) = -20*log10(norm(interp-raw)/norm(raw))

L,R = pCholtest(X)
println(snr(X,L*R'))
