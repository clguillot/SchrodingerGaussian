
#=
    Computes the local residual
        ∫_(0,h) ds |(i∂ₜ-H(t+s))(ζ₀(s)G0 + ζ₁(s)G1)
                        - ∑ᵣ(ζ₀(s)Gf0[r] + ζ₁(s)Gf1[r]))
                        - ∑ᵣ(ζ₀'(s)Gg0[r] + ζ₁'(s)Gg1[r]))|^2
            -   ∫_(0,h) ds |∑ᵣ(ζ₀(s)Gf0[r] + ζ₁(s)Gf1[r]))
                            + ∑ᵣ(ζ₀'(s)Gg0[r] + ζ₁'(s)Gg1[r]))|^2
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - G0, G1 are obtained by unpacking respectively X[1:6] and X[7:12]
        - H(t)g = apply_op(t, Gop, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_local_residual(t::T, h::T, apply_op, Gop::Gtype,
                                    Gf0::AbstractVector{<:GaussianWavePacket1D}, Gf1::AbstractVector{<:GaussianWavePacket1D},
                                    Gg0::AbstractVector{<:GaussianWavePacket1D}, Gg1::AbstractVector{<:GaussianWavePacket1D},
                                    X::AbstractVector{T1},
                                    ::Val{check_len}=Val(true)) where{T<:Real, Gtype, T1<:Real, check_len}
    
    if check_len && length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    
    G0 = unpack_gaussian_parameters(X, 1)
    G1 = unpack_gaussian_parameters(X, gaussian_param_size + 1)

    HG0 = apply_op(t, Gop, G0)
    HG1 = apply_op(t+h, Gop, G1)

    #Quadratic part

    #|i∂ₜ|^2
    S = 1/h * ((real(dot_L2(G0, G0)) + real(dot_L2(G1, G1))))
    S -= 2/h * real(dot_L2(G0, G1))

    #|H(t)|^2
    S += h/3 * (real(dot_L2(HG0, HG0)) + real(dot_L2(HG1, HG1)))
    S += h/3 * real(dot_L2(HG0, HG1))
    
    #-2*Re<i∂ₜ,H(t)> = -2*Im<∂ₜ,H(t)>
    S += imag(dot_L2(G0, HG0)) - imag(dot_L2(G1, HG1))
    S += imag(dot_L2(G0, HG1)) - imag(dot_L2(G1, HG0))

    #Linear part

    #-2*Re<(i∂ₜ-H(t)),f(t)>
    for g in Gf0
        #-2*Re<i∂ₜ,f(t)> = -2*Im<∂ₜ,f(t)>
        S += imag(dot_L2(G0, g))
        S -= imag(dot_L2(G1, g))

        #2*Re<H(t),f(t)>
        S += 2 * h/3 * real(dot_L2(HG0, g))
        S += h/3 * real(dot_L2(HG1, g))
    end
    for g in Gf1
        #-2*Re<i∂ₜ,f(t)> = -2*Im<∂ₜ,f(t)>
        S += imag(dot_L2(G0, g))
        S -= imag(dot_L2(G1, g))

        #2*Re<H(t),f(t)>
        S += h/3 * real(dot_L2(HG0, g))
        S += 2 * h/3 * real(dot_L2(HG1, g))
    end

    #-2*Re<(i∂ₜ-H(t)),g(t)>
    for g in Gg0
        #-2*Re<i∂ₜ,g(t)> = -2*Im<∂ₜ,g(t)>
        S -= 2 / h * imag(dot_L2(G0, g))
        S += 2 / h * imag(dot_L2(G1, g))

        #2*Re<H(t),g(t)>
        S -= real(dot_L2(HG0, g))
        S -= real(dot_L2(HG1, g))
    end
    for g in Gg1
        #-2*Re<i∂ₜ,g(t)> = -2*Im<∂ₜ,g(t)>
        S += 2 / h * imag(dot_L2(G0, g))
        S -= 2 / h * imag(dot_L2(G1, g))

        #2*Re<H(t),g(t)>
        S += real(dot_L2(HG0, g))
        S += real(dot_L2(HG1, g))
    end

    return S
end

#=
    Computes the local residual
        ∫_(0,h) ds <(i∂ₜ-H(t+s))(ζ₀(s)F0 + ζ₁(s)F1), (i∂ₜ-H(t+s))(ζ₀(s)G0 + ζ₁(s)G1)>
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - F0, F1 are obtained by unpacking respectively X1[1:6] and X1[7:12]
        - G0, G1 are obtained by unpacking respectively X2[1:6] and X2[7:12]
        - H(t)g = apply_op(t, Gop, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_local_residual_sesquilinear_part(t::T, h::T, apply_op, Gop::Gtype,
            X1::AbstractVector{T1}, X2::AbstractVector{T2},
            ::Val{check_len}=Val(true)) where{T<:Real, Gtype, T1<:Real, T2<:Real, check_len}
    
    if check_len && length(X1) != 2*gaussian_param_size
        throw(DimensionMismatch("X1 and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    if check_len && length(X2) != 2*gaussian_param_size
        throw(DimensionMismatch("X2 and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    
    F0 = unpack_gaussian_parameters(X1, 1)
    F1 = unpack_gaussian_parameters(X1, gaussian_param_size + 1)
    HF0 = apply_op(t, Gop, F0)
    HF1 = apply_op(t+h, Gop, F1)

    G0 = unpack_gaussian_parameters(X2, 1)
    G1 = unpack_gaussian_parameters(X2, gaussian_param_size + 1)
    HG0 = apply_op(t, Gop, G0)
    HG1 = apply_op(t+h, Gop, G1)

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
        - H(t)g = apply_op(t, Gop, g) for any gaussian wave packet g
=#
function schrodinger_gaussian_local_residual_linear_part(t::T, h::T, apply_op, Gop::Gtype,
            Gf0::AbstractVector{<:GaussianWavePacket1D}, Gf1::AbstractVector{<:GaussianWavePacket1D},
            X::AbstractVector{T1},
            ::Val{check_len}=Val(true)) where{T<:Real, Gtype, T1<:Real, check_len}
    
    if check_len && length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X and must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    
    G0 = unpack_gaussian_parameters(X, 1)
    G1 = unpack_gaussian_parameters(X, gaussian_param_size + 1)
    HG0 = apply_op(t, Gop, G0)
    HG1 = apply_op(t+h, Gop, G1)

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

#=

    GRADIENT

=#

mutable struct SchGaussianLocalGradientCFG
    gradient_cfg::ForwardDiff.GradientConfig
end
function SchGaussianLocalGradientCFG(X::Vector{T}) where{T<:Real}
    if length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end

    gradient_cfg = ForwardDiff.GradientConfig(x -> nothing, X, ForwardDiff.Chunk(2*gaussian_param_size))
    return SchGaussianLocalGradientCFG(gradient_cfg)
end

function schrodinger_gaussian_local_residual_gradient!(∇::Vector{T}, t::T, h::T, apply_op, Gop::Gtype,
                                            Gf0::AbstractVector{<:GaussianWavePacket1D}, Gf1::AbstractVector{<:GaussianWavePacket1D},
                                            Gg0::AbstractVector{<:GaussianWavePacket1D}, Gg1::AbstractVector{<:GaussianWavePacket1D},
                                            X::Vector{T},
                                            cfg::SchGaussianLocalGradientCFG=SchGaussianLocalGradientCFG(X)) where{T<:Real, Gtype}
    if length(∇) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(2*gaussian_param_size) but has size $(length(∇))"))
    end
    if length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a Vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end
    f(Y) = schrodinger_gaussian_local_residual(t, h, apply_op, Gop, Gf0, Gf1, Gg0, Gg1, Y)
    ForwardDiff.gradient!(∇, f, X, cfg.gradient_cfg, Val(false))
    return ∇
end