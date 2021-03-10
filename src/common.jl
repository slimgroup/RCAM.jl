function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, Core.eval(Main, Expr(:(=), nm, val)))
    end
end

