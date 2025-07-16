#=
    Computes the local residual
        ∫₍₀,ₜ₎ ds <i∂ₜζₖ(s)G0-ζₖ(s)HG0, i∂ₜζₗ(s)G1-ζₗ(s)HG1>
    where
        - The (ζⱼ)ⱼ are the P1 finite element functions such that ζⱼ((l-1)h)=δⱼₗ
        - t = (Lt-1)*h
=#
function schrodinger_gaussian_cross_residual(h::Real, Lt::Int, k::Int, l::Int,
                G0::AbstractWavePacket{D}, G1::AbstractWavePacket{D}, HG0::AbstractWavePacket{D}, HG1::AbstractWavePacket{D}) where D
    TS = promote_type(core_type(G0), core_type(G1), core_type(HG0), core_type(HG1))
    S = zero(complex(TS))

    if abs(k - l) <= 1 && max(k, l) <= Lt && min(k, l) >= 1
        # <i∂ₜG0,i∂ₜG1>
        S += fe_k_factor(h, k, l) * dot_L2(G0, G1)

        # <HG0,HG1>
        S += fe_m_factor(h, k, l) * dot_L2(HG0, HG1)
        
        #=
            - <i∂ₜG0,HG1> - <HG0,i∂ₜG1>
                = i<∂ₜG0,HG1> - i<HG0,∂ₜG1>
        =#
        if k == l
            if k == 1
                S /= 2
                S += fe_l_factor(h, 0, 1) * im_unit_mul(dot_L2(G0, HG1))
                S -= fe_l_factor(h, 0, 1) * im_unit_mul(dot_L2(HG0, G1))
            elseif k == Lt
                S /= 2
                S += fe_l_factor(h, 1, 0) * im_unit_mul(dot_L2(G0, HG1))
                S -= fe_l_factor(h, 1, 0) * im_unit_mul(dot_L2(HG0, G1))
            end
        else
            S += fe_l_factor(h, k, l) * im_unit_mul(dot_L2(G0, HG1))
            S -= fe_l_factor(h, l, k) * im_unit_mul(dot_L2(HG0, G1))
        end
    end

    return S
end

#=
    Computes the local residual
        ∫₍₀,ₜ₎ ds |i∂ₜζₖ(s)G0-ζₖ(s)HG0|^2
    where
        - The (ζⱼ)ⱼ are the P1 finite element functions such that ζⱼ((l-1)h)=δⱼₗ
        - t = (Lt-1)*h
=#
function schrodinger_gaussian_square_residual(h::Real, Lt::Int, k::Int,
                G0::AbstractWavePacket{D}, HG0::AbstractWavePacket{D}) where D
    
    # |i∂ₜ|^2
    S = fe_k_factor(h, 0, 0) * norm2_L2(G0)

    # |H(t)|^2
    S += fe_m_factor(h, 0, 0) * norm2_L2(HG0)

    # -2*Re<i∂ₜ,H(t)> = -2*Im<∂ₜ,H(t)>
    if k == 1
        S /= 2
        S -= 2 * fe_l_factor(h, 0, 1) * imag(dot_L2(G0, HG0))
    elseif k == Lt
        S /= 2
        S -= 2 * fe_l_factor(h, 1, 0) * imag(dot_L2(G0, HG0))
    end

    return S
end

#=
    Computes the gradient of
        ∫_(a,b) dt ||^2
    with respect to the variables X[(k-1)*psize + 1 : k*psize],
    where
    ...
=#
function schrodinger_gaussian_residual_local_gradient(::Type{Gtype}, a::T, b::T, Lt::Int, k::Int,
                                        apply_op, Gf::AbstractMatrix{<:AbstractWavePacket{D}}, Gg::AbstractMatrix{<:AbstractWavePacket{D}},
                                        X::AbstractVector{T}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)
    if !(1 <= k <= Lt)
        throw(BoundsError("k is equal to $k but must be between 1 and Lt-1=$(Lt-1)"))
    end
    if length(X) != Lt*psize
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    
    Xk = SVector{psize}(@view X[(k-1)*psize + 1 : k*psize])
    h = (b-a)/(Lt-1)
    t = a + (k-1)*h


    Gz = zero(Gtype)
    Gm1 = (k > 1) ? unpack_gaussian_parameters(Gtype, X, (k-2)*psize + 1) : Gz
    Gp1 = (k < Lt) ? unpack_gaussian_parameters(Gtype, X, k*psize + 1) : Gz
    HGm1 = apply_op(t-h, Gm1)
    HGp1 = apply_op(t+h, Gp1)

    function f(Y)
        G = unpack_gaussian_parameters(Gtype, Y)
        HG = apply_op(t, G)
    
        # Quadratic part
        S = schrodinger_gaussian_square_residual(h, Lt, k, G, HG)
        S += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k-1, G, Gm1, HG, HGm1))
        S += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k+1, G, Gp1, HG, HGp1))
    
        # Linear part
        g(l) = real(schrodinger_gaussian_cross_residual(h, Lt, k, l, G, WavePacketSum{D}(@view Gg[:, l]), HG, WavePacketSum{D}(@view Gf[:, l])))
        S -= 2 * sum(g, max(1,k-1):min(Lt,k+1))
    
        return S
    end
    
    return ForwardDiff.gradient(f, Xk)
end

#=
    Computes ∂ₓ₁∂ₓ₂E(X1, X2)
=#
function schrodinger_gaussian_residual_local_metric(::Type{Gtype}, a::T, b::T, Lt::Int, k::Int, l::Int,
                apply_op, X1::AbstractVector{T}, X2::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X1) != Lt*psize
        throw(DimensionMismatch("X1 must be a vector of size $(Lt*psize) but has size $(length(X1))"))
    end
    if length(X2) != Lt*psize
        throw(DimensionMismatch("X2 must be a vector of size $(Lt*psize) but has size $(length(X2))"))
    end

    h = (b-a)/(Lt-1)
    tk = a + k*h
    tl = a + l*h
    U1k = SVector{psize}(@view X1[(k-1)*psize + 1 : k*psize])
    U2l = SVector{psize}(@view X2[(l-1)*psize + 1 : l*psize])

    # function f(Y1, Y2)
    #     G1 = unpack_gaussian_parameters(Gtype, Y1)
    #     G2 = unpack_gaussian_parameters(Gtype, Y2)
    #     S = fe_k_factor(h, k, l) * real(dot_L2(G1, G2))
    #     if k == 1 || k == Lt
    #         S /= 2
    #     end
    #     return S
    # end
    function f(Y1, Y2)
        G1 = unpack_gaussian_parameters(Gtype, Y1)
        HG1 = apply_op(tk, G1)
        G2 = unpack_gaussian_parameters(Gtype, Y2)
        HG2 = apply_op(tl, G2)
        return real(schrodinger_gaussian_cross_residual(h, Lt, k, l, G1, G2, HG1, HG2))
    end
    ∇₁f(Y1, Y2) = ForwardDiff.gradient(Z -> f(Z, Y2), Y1)
    return ForwardDiff.jacobian(Z -> ∇₁f(U1k, Z), U2l)
end