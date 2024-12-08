
#=

    RESIDUAL

=#

#=
    Computes the residual |G1 - ∑ₖ Ginit[k]|^2 + (b-a) * ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For 1≤k≤Lt Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t)g = apply_op(t, Gop, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_residual(a::T, b::T, Lt::Int,
                        Ginit::AbstractVector{<:GaussianWavePacket1D}, apply_op, Gop::Gtype,
                        Gf::AbstractMatrix{<:GaussianWavePacket1D}, Gg::AbstractMatrix{<:GaussianWavePacket1D},
                        X::AbstractVector{T1}) where{T<:Real, Gtype, T1<:Real}
    
    if length(X) != gaussian_param_size * Lt
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end

    val = zero(promote_type(T, T1))
    h = (b - a) / (Lt - 1)

    #PDE residual
    for k=1:Lt-1
        t = a + (k-1)*h
        val += @views schrodinger_gaussian_local_residual(t, h, apply_op, Gop,
                    Gf[:, k], Gf[:, k+1], Gg[:, k], Gg[:, k+1],
                    X[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size],
                    Val(false))
    end
    val *= (b - a)

    #Initial condition
    val += @views gaussian_approx_residual(X[1 : gaussian_param_size], Ginit)

    return val
end

#=

=#
function schrodinger_gaussian_residual_sesquilinear_part(a::T, b::T, Lt::Int, apply_op, Gop::Gtype,
                        X1::AbstractVector{T1}, X2::AbstractVector{T2}) where{T<:Real, Gtype, T1<:Real, T2<:Real}
    
    if length(X1) != gaussian_param_size * Lt
        throw(DimensionMismatch("X1 must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X1))"))
    end
    if length(X2) != gaussian_param_size * Lt
        throw(DimensionMismatch("X2 must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X2))"))
    end

    val = zero(promote_type(T, T1, T2))
    h = (b - a) / (Lt - 1)

    #PDE residual
    for k=1:Lt-1
        t = a + (k-1)*h
        @views Y1 = X1[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size]
        @views Y2 = X2[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size]
        @views val += schrodinger_gaussian_local_residual_sesquilinear_part(t, h, apply_op, Gop, Y1, Y2, Val(false))
    end
    val *= (b - a)

    #Initial condition
    G1 = @views unpack_gaussian_parameters(X1[1 : gaussian_param_size])
    G2 = @views unpack_gaussian_parameters(X2[1 : gaussian_param_size])
    val += dot_L2(G1, G2)

    return val
end

#=

=#
function schrodinger_gaussian_residual_linear_part(a::T, b::T, Lt::Int,
                        Ginit::AbstractVector{<:GaussianWavePacket1D}, apply_op, Gop::Gtype,
                        Gf::AbstractMatrix{<:GaussianWavePacket1D},
                        X::AbstractVector{T1}) where{T<:Real, Gtype, T1<:Real}
    
    if length(X) != gaussian_param_size * Lt
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end

    val = zero(promote_type(T, T1))
    h = (b - a) / (Lt - 1)

    #PDE residual
    for k=1:Lt-1
        t = a + (k-1)*h
        @views Y = X[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size]
        @views val += schrodinger_gaussian_local_residual_linear_part(t, h, Gv,
                    Gf[:, k], Gf[:, k+1],
                    Y, Val(false))
    end
    val *= (b - a)

    #Initial condition
    G0 = @views unpack_gaussian_parameters(X[1 : gaussian_param_size])
    val = sum((-2 * dot_L2(g, G0) for g in Ginit); init=val)

    return val
end

#=

    GRADIENT

=#

mutable struct SchGaussianGradientCFG{T}
    Y0::Vector{T}
    Y::Vector{Vector{T}}
    fg0::Vector{T}
    fg::Vector{Vector{T}}
    cfg0::GaussianApproxGradientCFG
    cfg_gradient::Vector{<:SchGaussianLocalGradientCFG}
end
function SchGaussianGradientCFG(X::AbstractVector{T}) where{T<:Real}
    nt = Threads.nthreads()
    Y0 = zeros(T, gaussian_param_size)
    Y = [zeros(T, 2*gaussian_param_size) for _=1:nt]
    fg0 = zeros(T, gaussian_param_size)
    fg = [zeros(T, 2*gaussian_param_size) for _=1:nt]
    cfg0 = GaussianApproxGradientCFG(Y0)
    cfg_gradient = [SchGaussianLocalGradientCFG(Y[1]) for _=1:nt]
    return SchGaussianGradientCFG(Y0, Y, fg0, fg, cfg0, cfg_gradient)
end

function schrodinger_gaussian_gradient!(∇::AbstractVector{T},
                            a::T, b::T, Lt::Int, Ginit::AbstractVector{<:GaussianWavePacket1D}, apply_op, Gop::Gtype,
                            Gf::Matrix{<:GaussianWavePacket1D},
                            Gg::Matrix{<:GaussianWavePacket1D},
                            X::AbstractVector{T},
                            cfg::SchGaussianGradientCFG=SchGaussianGradientCFG(X)) where{T<:Real, Gtype}
    
    if length(X) != gaussian_param_size * Lt
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end
    if length(∇) != gaussian_param_size * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(∇))"))
    end

    #Sets A and ∇ to zero
    fill!(∇, zero(T))

    h = (b - a) / (Lt - 1)

    #PDE residual
    function loc_grad(k)
        #Recovers buffers
        kb = threadid()
        Y = cfg.Y[kb]
        fg = cfg.fg[kb]

        t = a + (k-1)*h
        @views copy!(Y, X[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size])

        #Gradient
        schrodinger_gaussian_local_residual_gradient!(fg, t, h, apply_op, Gop,
            (@view Gf[:, k]), (@view Gf[:, k+1]), (@view Gg[:, k]), (@view Gg[:, k+1]),
            Y, cfg.cfg_gradient[kb])
        @. @views ∇[(k-1) * gaussian_param_size + 1 : (k+1) * gaussian_param_size] += (b - a) * fg
    end
    #Odd part
    @threads :static for k=1:2:Lt-1
        loc_grad(k)
    end
    #Even part
    @threads :static for k=2:2:Lt-1
        loc_grad(k)
    end

    #Initial condition
    @views copy!(cfg.Y0, X[1 : gaussian_param_size])
    gaussian_approx_gradient!(cfg.fg0, Ginit, cfg.Y0, cfg.cfg0)
    @. @views ∇[1 : gaussian_param_size] += cfg.fg0

    return ∇
end

#=

    GRADIENT AND METRIC

=#

mutable struct SchGaussianGradientAndMetricCFG{T}
    Yk::Vector{Vector{T}}
    Yl::Vector{Vector{T}}
    Y::Vector{Vector{T}}
    fg0::Vector{T}
    fh0::Matrix{T}
    fg::Vector{Vector{T}}
    fh::Vector{Matrix{T}}
    cfg0::GaussianApproxGradientAndMetricCFG
    cfg_gradient::Vector{<:SchGaussianLocalGradientCFG}
    cfg_metric::Vector{<:GaussianApproxMetricTRHessCFG}
end
function SchGaussianGradientAndMetricCFG(::Type{T}) where{T<:Real}
    nt = Threads.nthreads()
    Yk = [zeros(T, gaussian_param_size) for _=1:nt]
    Yl = [zeros(T, gaussian_param_size) for _=1:nt]
    Y = [zeros(T, 2*gaussian_param_size) for _=1:nt]
    fg0 = zeros(T, gaussian_param_size)
    fh0 = zeros(T, gaussian_param_size, gaussian_param_size)
    fg = [zeros(T, 2*gaussian_param_size) for _=1:nt]
    fh = [zeros(T, gaussian_param_size, gaussian_param_size) for _=1:nt]
    cfg0 = GaussianApproxGradientAndMetricCFG(Yk[1])
    cfg_gradient = [SchGaussianLocalGradientCFG(Y[1]) for _=1:nt]
    cfg_metric = [GaussianApproxMetricTRHessCFG(Yk[1], Yl[1]) for _=1:nt]
    return SchGaussianGradientAndMetricCFG(Yk, Yl, Y, fg0, fh0, fg, fh, cfg0, cfg_gradient, cfg_metric)
end

#=
    Computes the gradient and hessian of the residual ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2 with respect to X
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For k≥1 Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t) = e^(-itΔ)Gv(x)e^(itΔ)
    - The resulting gradient and hessian are respectively stored into ∇ and A
    Returns ∇, A
=#
function schrodinger_gaussian_gradient_and_metric!(∇::AbstractVector{T}, A::BlockBandedMatrix{T},
                                        a::T, b::T, Lt::Int, Ginit::AbstractVector{<:GaussianWavePacket1D}, apply_op, Gop::Gtype,
                                        Gf::Matrix{<:GaussianWavePacket1D},
                                        Gg::Matrix{<:GaussianWavePacket1D},
                                        X::AbstractVector{T},
                                        cfg::SchGaussianGradientAndMetricCFG=SchGaussianGradientAndMetricCFG(T)) where{T<:Real, Gtype}
    
    if length(X) != gaussian_param_size * Lt
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end
    if length(∇) != gaussian_param_size * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(∇))"))
    end

    #Sets A and ∇ to zero
    fill!(∇, zero(T))
    fill!(A, zero(T))

    h = (b - a) / (Lt - 1)

    #=
        Returns ∫_(a,b) dt ζₖ'(t)ζₗ'(t)
    =#
    function fe_factor(k, l)
        if k==l==1 || k==l==Lt
            return 1/h
        elseif k==l
            return 2/h
        elseif abs(k - l) == 1
            return -1/h
        else
            return 0/h
        end
    end

    #PDE residual
    function loc_grad(k)
        #Recovers buffers
        kb = threadid()
        Y = cfg.Y[kb]
        fg = cfg.fg[kb]

        t = a + (k-1)*h
        @views Y .= X[(k-1)*gaussian_param_size + 1 : (k+1)*gaussian_param_size]

        #Gradient
        schrodinger_gaussian_local_residual_gradient!(fg, t, h, apply_op, Gop,
            (@view Gf[:, k]), (@view Gf[:, k+1]), (@view Gg[:, k]), (@view Gg[:, k+1]),
            Y, cfg.cfg_gradient[kb])
        @. @views ∇[(k-1) * gaussian_param_size + 1 : (k+1) * gaussian_param_size] += (b - a) * fg
    end
    #Odd part
    @threads :static for k=1:2:Lt-1
        loc_grad(k)
    end
    #Even part
    @threads :static for k=2:2:Lt-1
        loc_grad(k)
    end

    #PDE residual
    function loc_metric(k, l)
        #Recovers buffers
        kb = threadid()
        Yk = cfg.Yk[kb]
        Yl = cfg.Yl[kb]
        fh = cfg.fh[kb]

        @views Yk .= X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        @views Yl .= X[(l-1)*gaussian_param_size + 1 : l*gaussian_param_size]

        gaussian_approx_metric_topright_hessian!(fh, Yk, Yl, cfg.cfg_metric[kb])
        α = (b - a) * fe_factor(k, l)
        if k==l
            @views @. A[Block(k, k)] = α * (fh + fh') / 2
        else
            @views @. A[Block(k, l)] = α * fh
            @views @. A[Block(l, k)] = α * fh'
        end
    end
    for k=1:Lt
        for l=k:min(Lt,k+1)
            loc_metric(k, l)
        end
    end

    #Initial condition
    Y0 = cfg.Yk[1]
    @views copy!(Y0, X[1 : gaussian_param_size])
    gaussian_approx_gradient_and_metric!(cfg.fg0, cfg.fh0, Ginit, Y0, cfg.cfg0)
    @. @views ∇[1 : gaussian_param_size] += cfg.fg0
    @. @views A[Block(1, 1)] += (cfg.fh0 + cfg.fh0') / 2

    return ∇, A
end