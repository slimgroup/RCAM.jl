function GenDataSize(m,n,r)

## This script creates an sxs data matrix consisting of a discrezation of 4 
## 2-D Gaussian bumps. The resulting matrix is rank 4 (= number of bumps)

sc = 1

# Create regular grid
st = 1/n
X = 0:st:1-st
X = sc*X

sy = 1/m
Y = 0:sy:1-sy
Y = sc*Y

c = 50/sc^2

Reg = zeros(m,n)

# Bump 1
for j = 1:r 

a = round(rand(1),2)
b = round(rand(1),2)

    for k = 1:m
        for l = 1:n
            Reg[k,l] += dot(exp((-c*([sc*a - X[l]].^2))[1]),exp((-c*([sc*b - Y[k]].^2))[1]))
        end
    end
end

return Reg
end
export GenDataSize


