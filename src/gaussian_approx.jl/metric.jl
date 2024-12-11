
#=
    Returns real(dot_L2(G_X1, G_X2))
=#
function gaussian_approx_metric(X1::AbstractVector{T1}, X2::AbstractVector{T2}, ::Val{check_len}=Val(true)) where{T1<:Real, T2<:Real, check_len}
    if check_len && (length(X1) != gaussian_param_size || length(X2) != gaussian_param_size)
        throw(DimensionMismatch("X1 and X2 must be vectors of size $gaussian_param_size"))
    end

    G1 = unpack_gaussian_parameters(X1)
    G2 = unpack_gaussian_parameters(X2)

    return real(dot_L2(G1, G2))
end

#Metric config
mutable struct GaussianApproxMetricTRHessCFG{T}
    W::Vector{T}
    gradient_cfg::ForwardDiff.GradientConfig
    jacobian_cfg::ForwardDiff.JacobianConfig
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
function gaussian_approx_metric_topright_hessian!(Htr::Matrix{T}, X1::Vector{T}, X2::Vector{T}, cfg::GaussianApproxMetricTRHessCFG=GaussianApproxMetricTRHessCFG(X1, X2)) where{T<:Real}
    if size(Htr) != (gaussian_param_size, gaussian_param_size)
        throw(DimensionMismatch("Htr must be a square matrix of size $(gaussian_param_size)x$(gaussian_param_size) but has size $(size(Htr))"))
    end
    if length(X1) != gaussian_param_size
        throw(DimensionMismatch("X1 must be a vector of size $gaussian_param_size but has size $(length(X1))"))
    end
    if length(X2) != gaussian_param_size
        throw(DimensionMismatch("X2 must be a vector of size $gaussian_param_size but has size $(length(X2))"))
    end

    f(Y1, Y2) = gaussian_approx_metric(Y1, Y2, Val(false))
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1, cfg.gradient_cfg, Val(false))
    ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X1, Z), cfg.W, 
        X2, cfg.jacobian_cfg, Val(false))
    return Htr
end