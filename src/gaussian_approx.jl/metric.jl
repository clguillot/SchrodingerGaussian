
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
function GaussianApproxMetricTRHessCFG(X1::Vector{T}, X2::Vector{T}) where{T<:Real}
    if length(X1) != gaussian_param_size
        throw(DimensionMismatch("X1 must be a vector of size $gaussian_param_size but has size $(length(X1))"))
    end
    if length(X2) != gaussian_param_size
        throw(DimensionMismatch("X2 must be a vector of size $gaussian_param_size but has size $(length(X2))"))
    end
    
    W = zeros(T, gaussian_param_size)
    jacobian_cfg = ForwardDiff.JacobianConfig(x -> nothing, W, X1, ForwardDiff.Chunk(2))
    gradient_cfg = ForwardDiff.GradientConfig(jacobian_cfg, X2, ForwardDiff.Chunk(2))
    return GaussianApproxMetricTRHessCFG(W, gradient_cfg, jacobian_cfg)
end

#=
    Computes ∂ₓ₁∂ₓ₂E(X1, X2)
    where E(x1, x2) = gaussian_approx_metric(x1, x2)
=#
function gaussian_approx_metric_topright_hessian!(Htr::Matrix{T}, X1::Vector{T}, X2::Vector{T}, cfg=GaussianApproxMetricTRHessCFG(X1, X2)) where{T<:Real}
    if size(Htr) != (gaussian_param_size, gaussian_param_size)
        throw(DimensionMismatch("Htr must be a square matrix of size $(gaussian_param_size)x$(gaussian_param_size) but has size $(size(Htr))"))
    end
    if length(X1) != gaussian_param_size
        throw(DimensionMismatch("X1 must be a vector of size $gaussian_param_size but has size $(length(X1))"))
    end
    if length(X2) != gaussian_param_size
        throw(DimensionMismatch("X2 must be a vector of size $gaussian_param_size but has size $(length(X2))"))
    end

    function f(Y1, Y2)
        G1 = unpack_gaussian_parameters(Y1)
        G2 = unpack_gaussian_parameters(Y2)
        return gaussian_approx_metric(G1, G2)
    end
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1, cfg.gradient_cfg, Val(false))
    ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X1, Z), cfg.W, 
        X2, cfg.jacobian_cfg, Val(false))
    return Htr
end