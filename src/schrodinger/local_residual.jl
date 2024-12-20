#=
    Computes the local residual
        2*real ∫₍₀,ₕ₎ ds <i∂ₜζ₀(s)G0-ζ₀(s)HG0, i∂ₜζ₁(s)G1-ζ₁(s)HG1>
    where
        - The (ζₖ)ₖ are the P1 finite element functions such that ζₖ(lh)=δₖₗ
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_cross_residual(h::T,
                                        G0::AbstractWavePacket, G1::AbstractWavePacket,
                                        HG0::AbstractWavePacket, HG1::AbstractWavePacket) where{T<:Real}
    # 2*Re<i∂ₜG0,i∂ₜG1>
    S = 2 * fe_k_factor(h, 0, 1) * real(dot_L2(G0, G1))

    # 2*Re<HG0,HG1>
    S += 2 * fe_m_factor(h, 0, 1) * real(dot_L2(HG0, HG1))
    
    #=
        - 2*Re<i∂ₜG0,HG1> - 2*Re<HG0,i∂ₜG1>
         = - 2*Im<∂ₜG0,HG1> + 2*Im<HG0,∂ₜG1>
    =#
    S += -2 * fe_l_factor(h, 0, 1) * imag(dot_L2(G0, HG1))
    S += 2 * fe_l_factor(h, 1, 0) * imag(dot_L2(HG0, G1))

    return S
end

#=
    Computes the local residual
        ∫₍₋ₕ,ₕ₎ ds |i∂ₜζ₀(s)G0-ζ₀(s)HG0|^2    if K == 0
        ∫₍₀,ₕ₎ ds |i∂ₜζ₀(s)G0-ζ₀(s)HG0|^2    if K > 0
        ∫₍₋ₕ,₀₎ ds |i∂ₜζ₀(s)G0-ζ₀(s)HG0|^2    if K < 0
    where
        - The (ζₖ)ₖ are the P1 finite element functions such that ζₖ(lh)=δₖₗ
        - G0 is obtained by unpacking X0
        - HG0 is obtained by unpacking HX0
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_square_residual(h::T, G0::AbstractWavePacket, HG0::AbstractWavePacket,
                                        ::Val{K}=Val(0)) where{T<:Real, K}
    
    # |i∂ₜ|^2
    S = fe_k_factor(h, 0, 0) * norm2_L2(G0)

    # |H(t)|^2
    S += fe_m_factor(h, 0, 0) * norm2_L2(HG0)

    # -2*Re<i∂ₜ,H(t)> = -2*Im<∂ₜ,H(t)>
    if K > 0
        S /= 2
        S += -2 * fe_l_half_factor(h) * imag(dot_L2(G0, HG0))
    elseif K < 0
        S /= 2
        S += 2 * fe_l_half_factor(h) * imag(dot_L2(G0, HG0))
    end

    return S
end

#=
    Computes the local residual
        -2*real ∫₍₋ₕ,ₕ₎ ds <∑ᵣζₛ'(s)Wg[r] + ∑ᵣζₛ(s)Wf[r],(i∂ₜ-H(t))ζ₀(s)G0>   if K2 == 0
        -2*real ∫₍₀,ₕ₎ ds <∑ᵣζₛ'(s)Wg[r] + ∑ᵣζₛ(s)Wf[r],(i∂ₜ-H(t))ζ₀(s)G0>   if K2 > 0
        -2*real ∫₍₋ₕ,₀₎ ds <∑ᵣζₛ'(s)Wg[r] + ∑ᵣζₛ(s)Wf[r],(i∂ₜ-H(t))ζ₀(s)G0>   if K2 < 0
    where
        - s = sign(K1)
        - The (ζₖ)ₖ are the P1 finite element functions such that ζₖ(lh)=δₖₗ
        - G0 is obtained by unpacking X0
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_linear_residual(h::T,
                                        G0::AbstractWavePacket, HG0::AbstractWavePacket,
                                        Wf::AbstractVector{<:AbstractWavePacket},
                                        Wg::AbstractVector{<:AbstractWavePacket},
                                        ::Val{K1}, ::Val{K2}=Val(0)) where{T<:Real, K1, K2}
    
    s1 = sign(K1)
    s2 = sign(K2)

    S = zero(real(promote_type(T, eltype(G0), eltype(HG0), eltype(eltype(Wf)), eltype(eltype(Wg)))))

    if abs(s1 - s2) <= 1
        # -2*Re<Wf(t),(i∂ₜ-H(t))G0>
        for f in Wf
            # -2*Re<Wf(t),i∂ₜG0> = 2*Im<Wf(t),∂ₜG0>
            if s1 == 0 && s2 != 0
                S += 2 * s2 * fe_l_half_factor(h) * imag(dot_L2(f, G0))
            elseif s1 != 0
                S += 2 * fe_l_factor(h, 0, s1) * imag(dot_L2(f, G0))
            end

            # 2*Re<Wf(t),H(t)G0>
            if s1 == 0 && s2 != 0
                S += fe_m_factor(h, 0, s1) * real(dot_L2(f, HG0))
            else
                S += 2 * fe_m_factor(h, 0, s1) * real(dot_L2(f, HG0))
            end
        end

        #-2*Re<Wg(t),(i∂ₜ-H(t))G0>
        for g in Wg
            # -2*Re<Wg(t),i∂ₜG0> = 2*Im<Wg(t),∂ₜG0>
            if s1 == 0 && s2 != 0
                S += fe_k_factor(h, s1, 0) * imag(dot_L2(g, G0))
            else
                S += 2 * fe_k_factor(h, s1, 0) * imag(dot_L2(g, G0))
            end

            # 2*Re<Wg(t),H(t)G0>
            if s1 == 0 && s2 != 0
                S += 2 * s2 * fe_l_half_factor(h) * real(dot_L2(g, HG0))
            elseif s1 != 0
                S += 2 * fe_l_factor(h, s1, 0) * real(dot_L2(g, HG0))
            end
        end
    end

    return S
end

#=
    Computes
        ∫_(0,h) ds |(i∂ₜ-H(t))ζ₀(s)G0 + (i∂ₜ-H(t+h))ζ₁(s)G1
                        - ∑ᵣ(ζ₀(s)Gf[r,k] + ζ₁(s)Gf[r,k+1]))
                        - ∑ᵣ(ζ₀'(s)Gg[r,k] + ζ₁'(s)Gg[r,k+1]))|^2
            -   ∫_(0,h) ds |∑ᵣ(ζ₀(s)Gf[r,k] + ζ₁(s)Gf1[r,k+1]))
                            + ∑ᵣ(ζ₀'(s)Gg0[r,k] + ζ₁'(s)Gg1[r,k+1]))|^2
    where
    - The (ζₖ)ₖ are the P1 finite element functions such that ζₖ(lh)=δₖₗ
    - h = (b-a)/(Lt-1)
    - t = a + (k-1)*h
    - G0 is obtained by unpacking X0
    - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_elementary_residual(a::T, b::T, Lt::Int, k::Int,
                                        apply_op,
                                        Wf::AbstractMatrix{<:AbstractWavePacket},
                                        Wg::AbstractMatrix{<:AbstractWavePacket},
                                        X::AbstractVector{T1}) where{T<:Real, T1<:Real}
    if !(1 <= k <= Lt-1)
        throw(BoundsError("k is equal to $k but must be between 1 and Lt-1=$(Lt-1)"))
    end
    if length(X) != Lt*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end
    
    h = (b-a)/(Lt-1)
    t = a + (k-1)*h

    G0 = unpack_gaussian_parameters(X, (k-1)*gaussian_param_size + 1)
    G1 = unpack_gaussian_parameters(X, k*gaussian_param_size + 1)

    HG0 = apply_op(t, G0)
    HG1 = apply_op(t+h, G1)

    # Quadratic part
    S = schrodinger_gaussian_square_residual(h, G0, HG0, Val(1)) +
            schrodinger_gaussian_square_residual(h, G1, HG1, Val(-1))
    S += schrodinger_gaussian_cross_residual(h, G0, G1, HG0, HG1)

    # Linear part
    Wf0 = @view Wf[:, k]
    Wf1 = @view Wf[:, k+1]
    Wg0 = @view Wg[:, k]
    Wg1 = @view Wg[:, k+1]
    S += schrodinger_gaussian_linear_residual(h, G0, HG0, Wf0, Wg0, Val(0), Val(1)) +
            schrodinger_gaussian_linear_residual(h, G0, HG0, Wf1, Wg1, Val(1), Val(1))
    S += schrodinger_gaussian_linear_residual(h, G1, HG1, Wf0, Wg0, Val(-1), Val(-1)) +
            schrodinger_gaussian_linear_residual(h, G1, HG1, Wf1, Wg1, Val(0), Val(-1))
    
    return S
end

#=
    Computes the gradient of
        ∫_(a,b) dt ||^2
    with respect to the variables X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size],
    where
    ...
=#
mutable struct SchGaussianLocalGradientCFG2{T<:Real}
    X0::Vector{T}
    cfg_gradient::ForwardDiff.GradientConfig
end
function SchGaussianLocalGradientCFG2(Lt::Int, X::AbstractVector{T}) where{T<:Real}
    if length(X) != Lt*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(Lt*gaussian_param_size) but has size $(length(X))"))
    end
    X0 = zeros(T, gaussian_param_size)
    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X0, ForwardDiff.Chunk(gaussian_param_size))
    return SchGaussianLocalGradientCFG2(X0, cfg_gradient)
end
function schrodinger_gaussian_residual_local_gradient!(∇::AbstractVector{T}, a::T, b::T, Lt::Int, k::Int,
                                        apply_op,
                                        Gf::AbstractMatrix{<:AbstractWavePacket},
                                        Gg::AbstractMatrix{<:AbstractWavePacket},
                                        X::AbstractVector{T},
                                        cfg::SchGaussianLocalGradientCFG2=SchGaussianLocalGradientCFG2(Lt, X)) where{T<:Real}
    if !(1 <= k <= Lt)
        throw(BoundsError("k is equal to $k but must be between 1 and Lt-1=$(Lt-1)"))
    end
    if length(X) != Lt*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(gaussian_param_size * Lt) but has size $(length(X))"))
    end
    
    h = (b-a)/(Lt-1)
    t = a + (k-1)*h

    X0 = cfg.X0
    X0 .= @view X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]

    if k == 1
        G1 = unpack_gaussian_parameters(X, gaussian_param_size + 1)
        HG1 = apply_op(t+h, G1)

        function f_right(Y)
            G = unpack_gaussian_parameters(Y)
            HG = apply_op(t, G)

            # Quadratic part
            S = schrodinger_gaussian_square_residual(h, G, HG, Val(1))
            S += schrodinger_gaussian_cross_residual(h, G, G1, HG, HG1)
            
            # Linear part
            @unroll for s=0:1
                S += @views schrodinger_gaussian_linear_residual(h, G, HG, Gf[:, 1+s], Gg[:, 1+s], Val(s), Val(1))
            end

            return S
        end

        return ForwardDiff.gradient!(∇, f_right, X0, cfg.cfg_gradient, Val(false))

    elseif k == Lt
        Gm1 = unpack_gaussian_parameters(X, (Lt-2)*gaussian_param_size + 1)
        HGm1 = apply_op(t-h, Gm1)

        function f_left(Y)
            G = unpack_gaussian_parameters(Y)
            HG = apply_op(t, G)

            # Quadratic part
            S = schrodinger_gaussian_square_residual(h, G, HG, Val(-1))
            S += schrodinger_gaussian_cross_residual(h, Gm1, G, HGm1, HG)
            
            # Linear part
            @unroll for s=-1:0
                S += @views schrodinger_gaussian_linear_residual(h, G, HG, Gf[:, end+s], Gg[:, end+s], Val(s), Val(-1))
            end

            return S
        end

        return ForwardDiff.gradient!(∇, f_left, X0, cfg.cfg_gradient, Val(false))
    else
        Gm1 = unpack_gaussian_parameters(X, (k-2)*gaussian_param_size + 1)
        Gp1 = unpack_gaussian_parameters(X, k*gaussian_param_size + 1)
        HGm1 = apply_op(t-h, Gm1)
        HGp1 = apply_op(t+h, Gp1)

        function f_middle(Y)
            G = unpack_gaussian_parameters(Y)
            HG = apply_op(t, G)

            # Quadratic part
            S = schrodinger_gaussian_square_residual(h, G, HG, Val(0))
            S += schrodinger_gaussian_cross_residual(h, Gm1, G, HGm1, HG)
            S += schrodinger_gaussian_cross_residual(h, G, Gp1, HG, HGp1)

            # Linear part
            @unroll for s=-1:1
                S += @views schrodinger_gaussian_linear_residual(h, G, HG, Gf[:, k+s], Gg[:, k+s], Val(s), Val(0))
            end

            return S
        end

        return ForwardDiff.gradient!(∇, f_middle, X0, cfg.cfg_gradient, Val(false))
    end
end

#=
    Computes
        ∫₍₀,ₕ₎ ds |(i∂ₜ-H(t))ζ₀(s)G0 + (i∂ₜ-H(t+h))ζ₁(s)G1
                        - ∑ᵣ(ζ₀(s)Gf[r,0] + ζ₁(s)Gf[r,1]))
                        - ∑ᵣ(ζ₀'(s)Gg[r,0] + ζ₁'(s)Gg[r,1]))|²
            -   ∫₍₀,ₕ₎ ds |(i∂ₜ-H(t))ζ₀(s)G0
                            - ∑ᵣ(ζ₀(s)Gf[r,0] + ζ₁(s)Gf[r,1]))
                            - ∑ᵣ(ζ₀'(s)Gg[r,0] + ζ₁'(s)Gg[r,1]))|²
    where
    - The (ζₖ)ₖ are the P1 finite element functions such that ζₖ(lh)=δₖₗ
    - h = (b-a)/(Lt-1)
    - t = a + (k-1)*h
    - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_timestep_residual(t::T, h::T, G0::AbstractWavePacket,
                                        apply_op,
                                        Wf::AbstractMatrix{<:AbstractWavePacket},
                                        Wg::AbstractMatrix{<:AbstractWavePacket},
                                        X::AbstractVector{T1}) where{T<:Real, T1<:Real}
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size gaussian_param_size but has size $(length(X))"))
    end

    G1 = unpack_gaussian_parameters(X)

    HG0 = apply_op(t, G0)
    HG1 = apply_op(t+h, G1)

    # Quadratic part
    S = schrodinger_gaussian_square_residual(h, G1, HG1, Val(-1))
    S += schrodinger_gaussian_cross_residual(h, G0, G1, HG0, HG1)

    # Linear part
    Wf0 = @view Wf[:, 1]
    Wf1 = @view Wf[:, 2]
    Wg0 = @view Wg[:, 1]
    Wg1 = @view Wg[:, 2]
    S += schrodinger_gaussian_linear_residual(h, G1, HG1, Wf0, Wg0, Val(-1), Val(-1)) +
            schrodinger_gaussian_linear_residual(h, G1, HG1, Wf1, Wg1, Val(0), Val(-1))
    
    return S
end

#=

=#
mutable struct SchGaussianGradientTimeStepCFG{T<:Real}
    X0::Vector{T}
    cfg_gradient::ForwardDiff.GradientConfig
end
function SchGaussianGradientTimeStepCFG(X::AbstractVector{T}) where{T<:Real}
    X0 = zeros(T, gaussian_param_size)
    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X0, ForwardDiff.Chunk(gaussian_param_size))
    return SchGaussianGradientTimeStepCFG(X0, cfg_gradient)
end
function schrodinger_gaussian_timestep_residual_gradient!(∇::AbstractVector{T}, t::T, h::T, G0::AbstractWavePacket,
                                        apply_op,
                                        Wf::AbstractMatrix{<:AbstractWavePacket},
                                        Wg::AbstractMatrix{<:AbstractWavePacket},
                                        X::AbstractVector{T},
                                        cfg::SchGaussianGradientTimeStepCFG=SchGaussianGradientTimeStepCFG(X)) where{T<:Real}
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $gaussian_param_size but has size $(length(X))"))
    end

    X0 = cfg.X0
    X0 .= X

    HG0 = apply_op(t, G0)

    function f(Y)
        G = unpack_gaussian_parameters(Y)
        HG = apply_op(t, G)

        # Quadratic part
        S = schrodinger_gaussian_square_residual(h, G, HG, Val(-1))
        S += schrodinger_gaussian_cross_residual(h, G0, G, HG0, HG)
        
        # Linear part
        @unroll for s=-1:0
            S += @views schrodinger_gaussian_linear_residual(h, G, HG, Wf[:, 2+s], Wg[:, 2+s], Val(s), Val(-1))
        end

        return S
    end

    return ForwardDiff.gradient!(∇, f, X0, cfg.cfg_gradient, Val(false))
end

#=
    Computes the local residual
        ∫_(0,h) ds <(i∂ₜ-H(t+s))(ζ₀(s)F0 + ζ₁(s)F1), (i∂ₜ-H(t+s))(ζ₀(s)G0 + ζ₁(s)G1)>
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - F0, F1 are obtained by unpacking respectively X1[1:6] and X1[7:12]
        - G0, G1 are obtained by unpacking respectively X2[1:6] and X2[7:12]
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_local_residual_sesquilinear_part(t::T, h::T, apply_op,
            X1::AbstractVector{T1}, X2::AbstractVector{T2},
            ::Val{check_len}=Val(true)) where{T<:Real, T1<:Real, T2<:Real, check_len}
    
    if check_len && length(X1) != 2*gaussian_param_size
        throw(DimensionMismatch("X1 and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    if check_len && length(X2) != 2*gaussian_param_size
        throw(DimensionMismatch("X2 and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    
    F0 = unpack_gaussian_parameters(X1, 1)
    F1 = unpack_gaussian_parameters(X1, gaussian_param_size + 1)
    HF0 = apply_op(t, F0)
    HF1 = apply_op(t+h, F1)

    G0 = unpack_gaussian_parameters(X2, 1)
    G1 = unpack_gaussian_parameters(X2, gaussian_param_size + 1)
    HG0 = apply_op(t, G0)
    HG1 = apply_op(t+h, G1)

    #Sesquilinear part

    # <i∂ₜ,i∂ₜ>
    S = (dot_L2(F0, G0) + dot_L2(F1, G1)) / h
    S -= (dot_L2(F0, G1) + dot_L2(F1, G0)) / h

    # <H,H>
    S += (h / 3) * (dot_L2(HF0, HG0) + dot_L2(HF1, HG1))
    S += (h / 6) * (dot_L2(HF0, HG1) + dot_L2(HF1, HG0))

    # -<i∂ₜ,H> - <H,i∂ₜ> = i(<∂ₜ,H> - <H,∂ₜ>)
    S -= (1im / 2) * (dot_L2(F0, HG0) - dot_L2(F1, HG1))
    S -= (1im / 2) * (dot_L2(F0, HG1) - dot_L2(F1, HG0))
    S += (1im / 2) * (dot_L2(HF1, G0) - dot_L2(HF0, G1))
    S += (1im / 2) * (dot_L2(HF0, G0) - dot_L2(HF1, G1))

    return S
end

#=
    Computes the local residual
        -2 ∫_(0,h) ds <(i∂ₜ-H(t+s))(ζ₀(s)G0 + ζ₁(s)G1),
                        ∑ᵣ(ζ₀(s)Gf0[r] + ζ₁(s)Gf1[r]))>
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - G0, G1 are obtained by unpacking respectively X[1:6] and X[7:12]
        - H(t)g = apply_op(t, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_local_residual_linear_part(t::T, h::T, apply_op,
            Gf0::AbstractVector{<:GaussianWavePacket1D}, Gf1::AbstractVector{<:GaussianWavePacket1D},
            X::AbstractVector{T1},
            ::Val{check_len}=Val(true)) where{T<:Real, T1<:Real, check_len}
    
    if check_len && length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    
    G0 = unpack_gaussian_parameters(X, 1)
    G1 = unpack_gaussian_parameters(X, gaussian_param_size + 1)
    HG0 = apply_op(t, G0)
    HG1 = apply_op(t+h, G1)

    S = zero(Complex{promote_type(T, T1)})

    # -2<f(t),(i∂ₜ-H(t))>
    for g in Gf0
        # -2<f(t),i∂ₜ> = -2i<f(t),∂ₜ>
        S += 1im * (dot_L2(g, G0) - dot_L2(g, G1))

        # 2<f(t),H(t)>
        S += h/3 * (2 * dot_L2(g, HG0) + dot_L2(g, HG1))
    end
    for g in Gf1
        # -2<f(t),i∂ₜ> = -2i<f(t),∂ₜ>
        S += 1im * (dot_L2(g, G0) - dot_L2(g, G1))

        # 2<f(t),H(t)>
        S += h/3 * (dot_L2(g, HG0) + 2 * dot_L2(g, HG1))
    end

    return S
end