
#=
    Computes |G_X - ∑G_list[k]|^2 - |∑G_list[k]|^2
=#
function gaussian_approx_residual(G::AbstractWavePacket, G_list)
    return norm2_L2(G) - 2 * real(dot_L2(G, G_list))
end

# #=
#     Computes -2<G_X, ∑G_list[k]>
# =#
# function gaussian_approx_residual_linear_part(G::AbstractWavePacket, G_list)
    
#     N = zero(real(promote_type(core_type(G), core_type(G_list))))
    
#     for g in G_list
#         N -= 2 * real(dot_L2(g, G))
#     end

#     return -2 * sum(real(dot_L2(G, gk)) for gk in G_list)
# end

#=
    Returns |∑G_list[k]|^2
=#
function gaussian_approx_residual_constant_part(G_list)
    return norm2_L2(G_list)
end

#Gradient Config
mutable struct GaussianApproxGradientCFG{GC}
    cfg_gradient::GC
end
function GaussianApproxGradientCFG(::Type{Gtype}, X::Vector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize
        throw(DimensionMismatch("X must be a vector of size $psize but has size $(length(X))"))
    end

    cfg_gradient = ForwardDiff.GradientConfig(x -> nothing, X, ForwardDiff.Chunk(psize))
    return GaussianApproxGradientCFG(cfg_gradient)
end
#Gradient
function gaussian_approx_gradient!(::Type{Gtype}, ∇::Vector{T}, G_list,
                                    X::Vector{T},
                                    cfg=GaussianApproxGradientCFG(Gtype, X)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X) != psize
        throw(DimensionMismatch("X must be a vector of size $psize but has size $(length(X))"))
    end

    function f(Y)
        G = unpack_gaussian_parameters(Gtype, Y)
        gaussian_approx_residual(G, G_list)
    end
    ForwardDiff.gradient!(∇, f, X, cfg.cfg_gradient, Val(false))
    return ∇
end