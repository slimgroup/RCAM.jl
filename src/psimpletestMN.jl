using MAT
fid = matopen("../data/nstest_in.mat")
b = read(fid, "b")
N = read(fid, "N")
alt = read(fid, "alt")
r = read(fid, "r")
eta = read(fid, "eta")

fidX = matopen("../data/X_in.mat")
X_in = read(fidX,"X")


#include("../src/AltMinFrob.jl")
using DistributedArrays
@everywhere include("../src/dclrMN_sync.jl")

@time ownerLMN,ownerRMN = dclrMN(b+N, eta, convert(Int,r), convert(Int,alt))

# Gather L and R results
L = fetch(@spawnat ownerL Main.L)
R = fetch(@spawnat ownerR Main.R)

@everywhere include("../src/dclr.jl")
@time ownerL,ownerR = dclr(b, convert(Int,r), convert(Int,alt))
LMN = fetch(@spawnat ownerLMN Main.L)
RMN = fetch(@spawnat ownerRMN Main.R)
