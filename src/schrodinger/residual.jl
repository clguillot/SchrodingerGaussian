
#=

    RESIDUAL

=#

#=
    Computes the residual |G1 - ∑ₖ Ginit[k]|^2 + (b-a) * ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2 - ...
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For 1≤k≤Lt Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_residual(::Type{Gtype}, a::T, b::T, Lt::Int, Ginit::AbstractWavePacket{D},
                apply_op, Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket},
                X::AbstractVector{T1}) where{D, Gtype<:AbstractWavePacket, T<:Real, T1<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end

    res = zero(real(promote_type(core_type(Ginit), T, T1)))
    h = (b-a) / (Lt-1)

    Gk = unpack_gaussian_parameters(Gtype, X)
    HGk = apply_op(a, Gk)
    for k in 1:Lt
        # Single element part
        res += schrodinger_gaussian_square_residual(h, Lt, k, Gk, HGk)
        for l=max(1,k-1):min(Lt,k+1)
            res -= @views 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, l, Gk, WavePacketSum{D}(Gg[:, l]), HGk, WavePacketSum{D}(Gf[:, l])))
        end

        # Interaction part
        if k < Lt
            Gl = unpack_gaussian_parameters(Gtype, X, k*psize + 1)
            HGl = apply_op(a + k*h, Gl)
            res += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k+1, Gk, Gl, HGk, HGl))

            # Updating iterators
            Gk = Gl
            HGk = HGl
        end
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

    val = zero(complex(promote_type(T, T1, T2)))
    h = (b-a)/(Lt-1)

    #PDE residual
    for k=1:Lt
        t = a + (k-1)*h
        G0 = unpack_gaussian_parameters(Gtype, X1, (k-1)*psize + 1)
        HG0 = apply_op(t, G0)
        for l=max(1,k-1):min(Lt,k+1)
            G1 = unpack_gaussian_parameters(Gtype, X2, (l-1)*psize + 1)
            HG1 = apply_op(a + (l-1)*h, G1)
            val += schrodinger_gaussian_cross_residual(h, Lt, k, l, G0, G1, HG0, HG1)
        end
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
                Ginit::AbstractWavePacket{D}, apply_op,
                Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket},
                X::AbstractVector{T1}) where{D, Gtype<:AbstractWavePacket, T<:Real, T1<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end

    val = zero(complex(promote_type(T, T1)))
    h = (b-a)/(Lt-1)

    #PDE residual
    for k in 1:Lt
        Gk = unpack_gaussian_parameters(Gtype, X, (k-1)*psize + 1)
        HGk = apply_op(a + (k-1)*h, Gk)
        for l=max(1,k-1):min(Lt,k+1)
            val += @views schrodinger_gaussian_cross_residual(h, Lt, l, k, WavePacketSum{D}(Gg[:, l]), Gk, WavePacketSum{D}(Gf[:, l]), HGk)
        end
    end
    val *= (b - a)

    #Initial condition
    G0 = unpack_gaussian_parameters(Gtype, X)
    val += dot_L2(Ginit, G0)

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
#=
    
=#
function schrodinger_gaussian_gradient!(::Type{Gtype}, ∇::AbstractVector{T},
                a::T, b::T, Lt::Int, Ginit::AbstractWavePacket, apply_op,
                Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket}, X::AbstractVector{T},
                cfg=SchGaussianGradientCFG(Gtype, Lt, X)) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    if length(∇) != psize * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(psize * Lt) but has size $(length(∇))"))
    end

    h = (b-a)/(Lt-1)
    res_atomic = Atomic{T}(zero(T))
    #PDE residual
    @threads :static for k in 1:Lt
        # Residual
        Gk = unpack_gaussian_parameters(Gtype, X, (k-1)*psize + 1)
        HGk = apply_op(a + (k-1)*h, Gk)
        val = schrodinger_gaussian_square_residual(h, Lt, k, Gk, HGk)
        for l=max(1,k-1):min(Lt,k+1)
            val -= @views 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, l, Gk, WavePacketSum{D}(Gg[:, l]), HGk, WavePacketSum{D}(Gf[:, l])))
        end
        if k < Lt
            Gkp1 = unpack_gaussian_parameters(Gtype, X, k*psize + 1)
            HGkp1 = apply_op(a + k*h, Gkp1)
            val += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k+1, Gk, Gkp1, HGk, HGkp1))
        end
        atomic_add!(res_atomic, val)

        # Gradient
        kb = threadid()
        ∇loc = @view ∇[(k-1)*psize + 1 : k*psize]
        schrodinger_gaussian_residual_local_gradient!(Gtype, ∇loc, a, b, Lt, k, apply_op, Gf, Gg, X, cfg.cfg_gradient[kb])
        ∇loc .*= (b-a)
    end
    res = res_atomic[]
    res *= (b - a)

    #Initial condition
    # Residual
    res += gaussian_approx_residual(unpack_gaussian_parameters(Gtype, X), Ginit)
    # Gradient
    @views copy!(cfg.Y, X[1 : psize])
    gaussian_approx_gradient!(Gtype, cfg.fg, Ginit, cfg.Y, cfg.cfg0)
    @views ∇[1 : psize] .+= cfg.fg

    return res, ∇
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
    cfg_metric = [SchGaussianLocalMetricCFG(Gtype, Lt, X, X) for _=1:nt]
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
                                        a::T, b::T, Lt::Int, Ginit::AbstractWavePacket, apply_op,
                                        Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket},
                                        X::AbstractVector{T},
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

            @views Yk .= X[(k-1)*psize + 1 : k*psize]
            @views Yl .= X[(l-1)*psize + 1 : l*psize]

            schrodinger_gaussian_residual_local_metric!(Gtype, fh, a, b, Lt, k, l, apply_op, X, X, cfg.cfg_metric[kb])
            if k==l
                @views A[Block(k, l)] .= Symmetric(fh)
            else
                @views A[Block(k, l)] .= fh
                @views A[Block(l, k)] .= fh'
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