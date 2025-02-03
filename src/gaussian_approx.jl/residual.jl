
#=
    Computes |G_X - ∑G_list[k]|^2 - |∑G_list[k]|^2
=#
function gaussian_approx_residual(X::AbstractVector{T}, G_list::AbstractVector{TG}, ::Val{check_len}=Val(true)) where{T<:Real, check_len, TG<:AbstractWavePacket}
    if check_len && length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size"))
    end

    G = unpack_gaussian_parameters(X)

    N = zero(real(promote_type(core_type(G), core_type(TG))))

    # Quadratic part
    N += real(dot_L2(G, G))

    # Linear part
    for g in G_list
        N -= 2 * real(dot_L2(g, G))
    end

    return N
end

#=
    Computes -2<G_X, ∑G_list[k]>
=#
function gaussian_approx_residual_linear_part(X::AbstractVector{T}, G_list::AbstractVector{TG}, ::Val{check_len}=Val(true)) where{T<:Real, TG<:AbstractWavePacket, check_len}
    
    if check_len && length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size"))
    end

    G = unpack_gaussian_parameters(X)

    N = zero(real(promote_type(core_type(G), core_type(TG))))
    
    for g in G_list
        N -= 2 * real(dot_L2(g, G))
    end

    return -2 * sum(real(dot_L2(G, gk)) for gk in G_list)
end

#=
    Returns |∑G_list[k]|^2
=#
function gaussian_approx_residual_constant_part(G_list::AbstractVector{<:GaussianWavePacket1D})
    Lg = length(G_list)

    #Diagonal part
    N = sum(real(dot_L2(g, g)) for g in G_list)
    N += 2 * sum(real(dot_L2(G_list[k], G_list[l])) for k in 1:Lg for l in k+1:Lg; init=zero(N))

    return N
end

#Gradient Config
mutable struct GaussianApproxGradientCFG{GC}
    cfg_gradient::GC
end
function GaussianApproxGradientCFG(X::Vector{T}) where{T<:Real}
    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X, ForwardDiff.Chunk(gaussian_param_size))
    return GaussianApproxGradientCFG(cfg_gradient)
end
#Gradient
function gaussian_approx_gradient!(∇::Vector{T}, G_list::AbstractVector{<:GaussianWavePacket1D},
                                    X::Vector{T},
                                    cfg=GaussianApproxGradientCFG(X)) where{T<:Real}
    f(Y) = gaussian_approx_residual(Y, G_list, Val(false))
    ForwardDiff.gradient!(∇, f, X, cfg.cfg_gradient, Val(false))
    return ∇
end