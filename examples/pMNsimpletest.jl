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
fid = matopen("../data/X.mat")
d = read(fid)
X = d["X"]

function pMNtest(X)
# Choltest is just a wrapper for this script

# Params
r = 4 #Rank
alt = 5 #Iterations
perc = 0.8 #Percent of missing values
s1,s2 = size(X)

# Subsample
ind = randperm(s1*s2)
inds = ind[1:Int(round(perc*s1*s2))]
b = copy(X)
b[inds] .= 0.

#Gen Noise
N = rand(Float64,size(b))
noise = 0.5*norm(vec(X))*N/norm(vec(N))
noise[inds] .= 0.

eta = norm(vec(noise))/sqrt(size(X)[1])

# Interpolate
ownerL, ownerR = dclrMN(b+noise, eta, r, alt)

# Gather L and R results
L = fetch(@spawnat ownerL Main.L)
R = fetch(@spawnat ownerR Main.R)

return L,R 
end

snr(raw,interp) = -20*log10(norm(interp-raw)/norm(raw))

L,R = pMNtest(X)
println(snr(X,L*R'))
