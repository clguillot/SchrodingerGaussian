
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
                apply_op, Gf::AbstractMatrix{<:AbstractWavePacket{D}}, Gg::AbstractMatrix{<:AbstractWavePacket{D}},
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
    G0 = unpack_gaussian_parameters(Gtype, X)
    res += gaussian_approx_residual(G0, Ginit)

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
        Gk = unpack_gaussian_parameters(Gtype, X1, (k-1)*psize + 1)
        HGk = apply_op(t, Gk)
        for l=max(1,k-1):min(Lt,k+1)
            Gl = unpack_gaussian_parameters(Gtype, X2, (l-1)*psize + 1)
            HGl = apply_op(a + (l-1)*h, Gl)
            val += schrodinger_gaussian_cross_residual(h, Lt, k, l, Gk, Gl, HGk, HGl)
        end
    end
    val *= (b - a)

    #Initial condition
    G01 = unpack_gaussian_parameters(Gtype, X1)
    G02 = unpack_gaussian_parameters(Gtype, X2)
    val += dot_L2(G01, G02)

    return val
end

#=

=#
function schrodinger_gaussian_residual_linear_part(::Type{Gtype}, a::T, b::T, Lt::Int,
                Ginit::AbstractWavePacket{D}, apply_op,
                Gf::AbstractMatrix{<:AbstractWavePacket{D}}, Gg::AbstractMatrix{<:AbstractWavePacket{D}},
                X::AbstractVector{T1}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real, T1<:Real}
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
    X0 = SVector{psize}(@view X[1:psize])
    G0 = unpack_gaussian_parameters(Gtype, X0)
    val += dot_L2(Ginit, G0)

    return val
end

#=

    GRADIENT

=#

#=
    
=#
function schrodinger_gaussian_gradient!(::Type{Gtype}, ∇::AbstractVector{T},
                a::T, b::T, Lt::Int, Ginit::AbstractWavePacket{D}, apply_op,
                Gf::AbstractMatrix{<:AbstractWavePacket{D}}, Gg::AbstractMatrix{<:AbstractWavePacket{D}},
                X::AbstractVector{T}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
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
    @threads for k in 1:Lt
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
        ∇loc = @view ∇[(k-1)*psize + 1 : k*psize]
        ∇loc .= (b-a) * schrodinger_gaussian_residual_local_gradient(Gtype, a, b, Lt, k, apply_op, Gf, Gg, X)
    end
    res = res_atomic[]
    res *= (b - a)

    #Initial condition
    X0 = SVector{psize}(@view X[1:psize])
    G0 = unpack_gaussian_parameters(Gtype, X0)
    # Residual
    res += gaussian_approx_residual(G0, Ginit)
    # Gradient
    ∇0 = @view ∇[1 : psize]
    ∇0 .+= gaussian_approx_gradient(Gtype, Ginit, X0)

    return res, ∇
end

#=

    GRADIENT AND METRIC

=#

#=
    Computes the gradient and hessian of the residual ∫_(a,b) dt |(i∂ₜ-H(t)(∑ₖζₖ(t)Gk)|^2 with respect to X
        - (ζₖ)ₖ (k=0,...,Lt-1) are the Lt P1 finite element functions on (a, b)
        - For k≥1 Gk is obtained by unpacking X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        - H(t)g = apply_op(t, g)
    - The resulting gradient and hessian are respectively stored into ∇ and A
    Returns ∇, A
=#
function schrodinger_gaussian_gradient_and_metric!(::Type{Gtype}, ∇::AbstractVector{T}, A::BlockBandedMatrix{T},
                                        a::T, b::T, Lt::Int, Ginit::AbstractWavePacket{D}, apply_op,
                                        Gf::AbstractMatrix{<:AbstractWavePacket{D}}, Gg::AbstractMatrix{<:AbstractWavePacket{D}},
                                        X::AbstractVector{T}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize * Lt
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    if length(∇) != psize * Lt
        throw(DimensionMismatch("∇ must be a Vector of size $(psize * Lt) but has size $(length(∇))"))
    end

    # PDE residual
    @threads for k=1:Lt
        # Gradient
        ∇loc = @view ∇[(k-1)*psize + 1 : k*psize]
        ∇loc .= (b-a) * schrodinger_gaussian_residual_local_gradient(Gtype, a, b, Lt, k, apply_op, Gf, Gg, X)

        # Metric
        for l=k:min(Lt,k+1)
            Akl = schrodinger_gaussian_residual_local_metric(Gtype, a, b, Lt, k, l, apply_op, X, X)
            if k==l
                @views A[Block(k, l)] .= Symmetric(Akl)
            else
                @views A[Block(k, l)] .= Akl
                @views A[Block(l, k)] .= Akl'
            end
        end
    end

    #Initial condition
    X0 = SVector{psize}(@view X[1:psize])
    ∇0, A0 = gaussian_approx_gradient_and_metric(Gtype, Ginit, X0)
    @views ∇[1 : psize] .+= ∇0
    @views A[Block(1, 1)] .+= (A0' + A0) / 2

    return ∇, A
end