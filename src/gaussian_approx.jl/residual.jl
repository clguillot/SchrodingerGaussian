
#=
    Computes |G_X - ∑G_list[k]|^2 - |∑G_list[k]|^2
=#
function gaussian_approx_residual(X::AbstractVector{T}, G_list::AbstractVector{<:GaussianWavePacket1D}, ::Val{check_len}=Val(true)) where{T<:Real, check_len}
    if check_len && length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size"))
    end

    G = unpack_gaussian_parameters(X)
    return real(dot_L2(G, G)) - 2 * sum(real(dot_L2(G, gk)) for gk in G_list)
end

#=
    Computes -2<G_X, ∑G_list[k]>
=#
function gaussian_approx_residual_linear_part(X::AbstractVector{T}, G_list::AbstractVector{<:GaussianWavePacket1D}, ::Val{check_len}=Val(true)) where{T<:Real, check_len}
    
    if check_len && length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size"))
    end

    G = unpack_gaussian_parameters(X)
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
mutable struct GaussianApproxGradientCFG
    cfg_gradient::ForwardDiff.GradientConfig
end
function GaussianApproxGradientCFG(X::Vector{T}) where{T<:Real}
    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X, ForwardDiff.Chunk(gaussian_param_size))
    return GaussianApproxGradientCFG(cfg_gradient)
end
#Gradient
function gaussian_approx_gradient!(∇::Vector{T}, G_list::AbstractVector{<:GaussianWavePacket1D},
                                    X::Vector{T},
                                    cfg::GaussianApproxGradientCFG=GaussianApproxGradientCFG(X)) where{T<:Real}
    f(Y) = gaussian_approx_residual(Y, G_list, Val(false))
    ForwardDiff.gradient!(∇, f, X, cfg.cfg_gradient, Val(false))
    return ∇
end

#=

    HESSIAN

=#

#Hessian Config
mutable struct GaussianApproxHessianCFG
    cfg_hessian::ForwardDiff.HessianConfig
end
function GaussianApproxHessianCFG(diff_result::DiffResults.MutableDiffResult{2, T, Tuple{Vector{T}, Matrix{T}}},
                                X::Vector{T}) where{T<:Real}
    cfg_hessian = ForwardDiff.HessianConfig(x -> nothing, diff_result, X, ForwardDiff.Chunk(gaussian_param_size))
    return GaussianApproxHessianCFG(cfg_hessian)
end
#Gradient
function gaussian_approx_hessian!(diff_result::DiffResults.MutableDiffResult{2, T, Tuple{Vector{T}, Matrix{T}}},
                                    G_list::AbstractVector{<:GaussianWavePacket1D},
                                    X::Vector{T},
                                    cfg::GaussianApproxHessianCFG=GaussianApproxHessianCFG(X)) where{T<:Real}
    f(Y) = gaussian_approx_residual(Y, G_list, Val(false))
    ForwardDiff.hessian!(diff_result, f, X, cfg.cfg_hessian, Val(false))
    return diff_result
end