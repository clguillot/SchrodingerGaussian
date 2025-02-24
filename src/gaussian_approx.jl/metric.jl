
#=
    Returns real(dot_L2(G_X1, G_X2))
=#
function gaussian_approx_metric(G1::AbstractWavePacket, G2::AbstractWavePacket)
    return real(dot_L2(G1, G2))
end

#Metric config
mutable struct GaussianApproxMetricTRHessCFG{T, GC, JC}
    W::Vector{T}
    gradient_cfg::GC
    jacobian_cfg::JC
end
function GaussianApproxMetricTRHessCFG(::Type{Gtype}, X1::Vector{T}, X2::Vector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if length(X1) != psize
        throw(DimensionMismatch("X1 must be a vector of size $psize but has size $(length(X1))"))
    end
    if length(X2) != psize
        throw(DimensionMismatch("X2 must be a vector of size $psize but has size $(length(X2))"))
    end
    
    W = zeros(T, psize)
    jacobian_cfg = ForwardDiff.JacobianConfig(x -> nothing, W, X1, ForwardDiff.Chunk(2))
    gradient_cfg = ForwardDiff.GradientConfig(jacobian_cfg, X2, ForwardDiff.Chunk(2))
    return GaussianApproxMetricTRHessCFG(W, gradient_cfg, jacobian_cfg)
end

#=
    Computes ∂ₓ₁∂ₓ₂E(X1, X2)
    where E(x1, x2) = gaussian_approx_metric(x1, x2)
=#
function gaussian_approx_metric_topright_hessian!(::Type{Gtype}, Htr::Matrix{T}, X1::Vector{T}, X2::Vector{T},
                                                    cfg=GaussianApproxMetricTRHessCFG(Gtype, X1, X2)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if size(Htr) != (psize, psize)
        throw(DimensionMismatch("Htr must be a square matrix of size $(psize)x$(psize) but has size $(size(Htr))"))
    end
    if length(X1) != psize
        throw(DimensionMismatch("X1 must be a vector of size $psize but has size $(length(X1))"))
    end
    if length(X2) != psize
        throw(DimensionMismatch("X2 must be a vector of size $psize but has size $(length(X2))"))
    end

    function f(Y1, Y2)
        G1 = unpack_gaussian_parameters(Gtype, Y1)
        G2 = unpack_gaussian_parameters(Gtype, Y2)
        return gaussian_approx_metric(G1, G2)
    end
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1, cfg.gradient_cfg, Val(false))
    ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X1, Z), cfg.W, 
        X2, cfg.jacobian_cfg, Val(false))
    return Htr
end