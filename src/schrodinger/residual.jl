
#=

    RESIDUAL

=#

#=
    Computes the residual |G1 - ∑ₖ Ginit[k]|^2 + (b-a) * ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For 1≤k≤Lt Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_residual(::Type{Gtype}, a::T, b::T, Lt::Int, Ginit::AbstractVector,
                apply_op, Gf, Gg, X::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end

    #PDE residual
    res = schrodinger_gaussian_elementary_residual(Gtype, a, b, Lt, 1, apply_op, Gf, Gg, X)
    for k in 2:Lt-1
        res += schrodinger_gaussian_elementary_residual(Gtype, a, b, Lt, k, apply_op, Gf, Gg, X)
    end
    res *= (b - a)

    #Initial condition
    res += gaussian_approx_residual(unpack_gaussian_parameters(Gtype, X), Ginit)

    return res
end

#=

=#
function schrodinger_gaussian_residual_sesquilinear_part(::Type{Gtype}, a::T, b::T, Lt::Int, apply_op,
                X1::AbstractVector{T1}, X2::AbstractVector{T2}) where{Gtype<:AbstractWavePacket, T<:Real, T1<:Real, T2<:Real}
    psize = param_size(Gtype)
    if length(X1) != psize * Lt
        throw(DimensionMismatch("X1 must be a Vector of size $(psize * Lt) but has size $(length(X1))"))
    end
    if length(X2) != psize * Lt
        throw(DimensionMismatch("X2 must be a Vector of size $(psize * Lt) but has size $(length(X2))"))
    end

    val = zero(promote_type(T, T1, T2))
    h = (b-a)/(Lt-1)

    #PDE residual
    for k=1:Lt-1
        t = a + (k-1)*h
        F0 = unpack_gaussian_parameters(Gtype, X1, (k-1)*psize + 1)
        F1 = unpack_gaussian_parameters(Gtype, X1, k*psize + 1)
        G0 = unpack_gaussian_parameters(Gtype, X2, (k-1)*psize + 1)
        G1 = unpack_gaussian_parameters(Gtype, X2, k*psize + 1)
        val += schrodinger_gaussian_local_residual_sesquilinear_part(t, h, apply_op, F0, F1, G0, G1)
    end
    val *= (b - a)

    #Initial condition
    G1 = unpack_gaussian_parameters(Gtype, X1)
    G2 = unpack_gaussian_parameters(Gtype, X2)
    val += dot_L2(G1, G2)

    return val
end

#=

=#
function schrodinger_gaussian_residual_linear_part(::Type{Gtype}, a::T, b::T, Lt::Int,
                Ginit::AbstractVector{<:GaussianWavePacket},
                apply_op, Gf, X::AbstractVector{T1}) where{Gtype<:AbstractWavePacket, T<:Real, T1<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end

    val = zero(promote_type(T, T1))
    h = (b - a) / (Lt - 1)

    #PDE residual
    for k=1:Lt-1
        t = a + (k-1)*h
        @views Y = X[(k-1)*psize + 1 : (k+1)*psize]
        @views val += schrodinger_gaussian_local_residual_linear_part(Gtype, t, h, apply_op,
                    Gf[:, k], Gf[:, k+1], Y, Val(false))
    end
    val *= (b - a)

    #Initial condition
    G0 = unpack_gaussian_parameters(Gtype, X)
    val -= 2 * dot_L2(Ginit, G0)

    return val
end

#=

    GRADIENT

=#

mutable struct SchGaussianGradientCFG{T<:Real, CG0, CGloc}
    Y::Vector{T}
    fg::Vector{T}
    cfg0::CG0
    cfg_gradient::Vector{CGloc}
end
function SchGaussianGradientCFG(::Type{Gtype}, Lt::Int, X::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    nt = nthreads()
    psize = param_size(Gtype)
    Y = zeros(T, psize)
    fg = zeros(T, psize)
    cfg0 = GaussianApproxGradientCFG(Gtype, Y)
    cfg_gradient = [SchGaussianLocalGradientCFG(Gtype, Lt, X) for _=1:nt]
    return SchGaussianGradientCFG(Y, fg, cfg0, cfg_gradient)
end
function schrodinger_gaussian_gradient!(::Type{Gtype}, ∇::AbstractVector{T},
                a::T, b::T, Lt::Int, Ginit::AbstractVector,
                apply_op, Gf, Gg, X::AbstractVector{T},
                cfg=SchGaussianGradientCFG(Gtype, Lt, X)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    if length(∇) != psize * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(psize * Lt) but has size $(length(∇))"))
    end

    #PDE residual
    @threads :static for k in 1:Lt
        kb = threadid()

        ∇loc = @view ∇[(k-1)*psize + 1 : k*psize]
        schrodinger_gaussian_residual_local_gradient!(Gtype, ∇loc, a, b, Lt, k, apply_op, Gf, Gg, X, cfg.cfg_gradient[kb])
        ∇loc .*= (b-a)
    end

    #Initial condition
    @views copy!(cfg.Y, X[1 : psize])
    gaussian_approx_gradient!(Gtype, cfg.fg, Ginit, cfg.Y, cfg.cfg0)
    @views ∇[1 : psize] .+= cfg.fg

    return ∇
end

#=

    GRADIENT AND METRIC

=#

mutable struct SchGaussianGradientAndMetricCFG{T<:Real, CG0, CG, CM}
    Yk::Vector{Vector{T}}
    Yl::Vector{Vector{T}}
    Y::Vector{Vector{T}}
    fg0::Vector{T}
    fh0::Matrix{T}
    fg::Vector{Vector{T}}
    fh::Vector{Matrix{T}}
    cfg0::CG0
    cfg_gradient::Vector{CG}
    cfg_metric::Vector{CM}
end
function SchGaussianGradientAndMetricCFG(::Type{Gtype}, Lt::Int, X::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    nt = Threads.nthreads()
    psize = param_size(Gtype)
    Yk = [zeros(T, psize) for _=1:nt]
    Yl = [zeros(T, psize) for _=1:nt]
    Y = [zeros(T, 2*psize) for _=1:nt]
    fg0 = zeros(T, psize)
    fh0 = zeros(T, psize, psize)
    fg = [zeros(T, 2*psize) for _=1:nt]
    fh = [zeros(T, psize, psize) for _=1:nt]
    cfg0 = GaussianApproxGradientAndMetricCFG(Gtype, Yk[1])
    cfg_gradient = [SchGaussianLocalGradientCFG(Gtype, Lt, X) for _=1:nt]
    cfg_metric = [GaussianApproxMetricTRHessCFG(Gtype, Yk[1], Yl[1]) for _=1:nt]
    return SchGaussianGradientAndMetricCFG(Yk, Yl, Y, fg0, fh0, fg, fh, cfg0, cfg_gradient, cfg_metric)
end

#=
    Computes the gradient and hessian of the residual ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2 with respect to X
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For k≥1 Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t)g = apply_op(t, g)
    - The resulting gradient and hessian are respectively stored into ∇ and A
    Returns ∇, A
=#
function schrodinger_gaussian_gradient_and_metric!(::Type{Gtype}, ∇::AbstractVector{T}, A::BlockBandedMatrix{T},
                                        a::T, b::T, Lt::Int, Ginit::AbstractVector, apply_op,
                                        Gf, Gg, X::AbstractVector{T},
                                        cfg=SchGaussianGradientAndMetricCFG(Gtype, Lt, X)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    if length(∇) != psize * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(psize * Lt) but has size $(length(∇))"))
    end

    # PDE residual
    @threads :static for k=1:Lt
        kb = threadid()

        # Gradient
        ∇loc = @view ∇[(k-1)*psize + 1 : k*psize]
        schrodinger_gaussian_residual_local_gradient!(Gtype, ∇loc, a, b, Lt, k, apply_op, Gf, Gg, X, cfg.cfg_gradient[kb])
        ∇loc .*= (b-a)

        # Metric
        for l=k:min(Lt,k+1)
            Yk = cfg.Yk[kb]
            Yl = cfg.Yl[kb]
            fh = cfg.fh[kb]
            h = (b-a)/(Lt-1)

            @views Yk .= X[(k-1)*psize + 1 : k*psize]
            @views Yl .= X[(l-1)*psize + 1 : l*psize]

            gaussian_approx_metric_topright_hessian!(Gtype, fh, Yk, Yl, cfg.cfg_metric[kb])
            α = (b - a) * fe_k_factor(h, k, l)
            if k==l && (k==1 || k==Lt)
                @views @. A[Block(k, k)] = α / 4 * (fh + fh')
            elseif k==l
                @views @. A[Block(k, k)] = α / 2 * (fh + fh')
            else
                @views @. A[Block(k, l)] = α * fh
                @views @. A[Block(l, k)] = α * fh'
            end
        end
    end

    #Initial condition
    Y0 = cfg.Yk[1]
    @views copy!(Y0, X[1 : psize])
    gaussian_approx_gradient_and_metric!(Gtype, cfg.fg0, cfg.fh0, Ginit, Y0, cfg.cfg0)
    @. @views ∇[1 : psize] += cfg.fg0
    @. @views A[Block(1, 1)] += (cfg.fh0 + cfg.fh0') / 2

    return ∇, A
end