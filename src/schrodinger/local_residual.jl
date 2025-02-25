#=
    Computes the local residual
        ∫₍₀,ₜ₎ ds <i∂ₜζₖ(s)G0-ζₖ(s)HG0, i∂ₜζₗ(s)G1-ζₗ(s)HG1>
    where
        - The (ζⱼ)ⱼ are the P1 finite element functions such that ζⱼ((l-1)h)=δⱼₗ
        - t = (Lt-1)*h
=#
function schrodinger_gaussian_cross_residual(h::Real, Lt::Int, k::Int, l::Int, G0, G1, HG0, HG1)
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
function schrodinger_gaussian_square_residual(h::Real, Lt::Int, k::Int, G0, HG0)
    
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
mutable struct SchGaussianLocalGradientCFG{T<:Real, CG}
    X0::Vector{T}
    cfg_gradient::CG
end
function SchGaussianLocalGradientCFG(::Type{Gtype}, Lt::Int, X::AbstractVector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != Lt*psize
        throw(DimensionMismatch("X must be a Vector of size $(Lt*psize) but has size $(length(X))"))
    end
    X0 = zeros(T, psize)
    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X0, ForwardDiff.Chunk(2))
    return SchGaussianLocalGradientCFG(X0, cfg_gradient)
end
function schrodinger_gaussian_residual_local_gradient!(::Type{Gtype}, ∇::AbstractVector{T}, a::T, b::T, Lt::Int, k::Int,
                                        apply_op, Gf, Gg, X::AbstractVector{T},
                                        cfg=SchGaussianLocalGradientCFG(Gtype, Lt, X)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if !(1 <= k <= Lt)
        throw(BoundsError("k is equal to $k but must be between 1 and Lt-1=$(Lt-1)"))
    end
    if length(X) != Lt*psize
        throw(DimensionMismatch("X must be a Vector of size $(psize * Lt) but has size $(length(X))"))
    end
    
    h = (b-a)/(Lt-1)
    t = a + (k-1)*h

    X0 = cfg.X0
    X0 .= @view X[(k-1)*psize + 1 : k*psize]
    Gz = zero(unpack_gaussian_parameters(Gtype, X, (k-1)*psize + 1))

    Gm1 = (k > 1) ? unpack_gaussian_parameters(Gtype, X, (k-2)*psize + 1) : Gz
    Gp1 = (k < Lt) ? unpack_gaussian_parameters(Gtype, X, k*psize + 1) : Gz
    HGm1 = apply_op(t-h, Gm1)
    HGp1 = apply_op(t+h, Gp1)

    function f(Y, t, h, Lt, k, apply_op, Gm1, Gp1, HGm1, HGp1, Gf, Gg)
        G = unpack_gaussian_parameters(Gtype, Y)
        HG = apply_op(t, G)

        # Quadratic part
        S = schrodinger_gaussian_square_residual(h, Lt, k, G, HG)
        S += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k-1, G, Gm1, HG, HGm1))
        S += 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, k+1, G, Gp1, HG, HGp1))

        # Linear part
        for l=max(1,k-1):min(Lt,k+1)
            S -= @views 2 * real(schrodinger_gaussian_cross_residual(h, Lt, k, l, G, Gg[:, l], HG, Gf[:, l]))
        end

        return S
    end

    g(Y) = f(Y, t, h, Lt, k, apply_op, Gm1, Gp1, HGm1, HGp1, Gf, Gg)
    return ForwardDiff.gradient!(∇, g, X0, cfg.cfg_gradient, Val(false))
end