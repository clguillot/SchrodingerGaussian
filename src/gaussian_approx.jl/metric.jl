
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
function GaussianApproxMetricTRHessCFG(X::Vector{T}) where{T<:Real}
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size but has size $(length(X))"))
    end
    
    W = zeros(T, gaussian_param_size)
    jacobian_cfg = ForwardDiff.JacobianConfig(x -> nothing, W, X, ForwardDiff.Chunk(gaussian_param_size))
    gradient_cfg = ForwardDiff.GradientConfig(jacobian_cfg, X, ForwardDiff.Chunk(gaussian_param_size))
    return GaussianApproxMetricTRHessCFG(W, gradient_cfg, jacobian_cfg)
end

#Metric
function gaussian_approx_metric_topright_hessian!(Htr::Matrix{T}, X::Vector{T}, cfg::GaussianApproxMetricTRHessCFG=GaussianApproxMetricTRHessCFG(X)) where{T<:Real}
    if size(Htr) != (gaussian_param_size, gaussian_param_size)
        throw(DimensionMismatch("Htr must be a square matrix of size $(gaussian_param_size)x$(gaussian_param_size) but has size $(size(Htr))"))
    end
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size but has size $(length(X))"))
    end

    f(Y1, Y2) = gaussian_approx_metric(Y1, Y2, Val(false))
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1, cfg.gradient_cfg, Val(false))
    ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X, Z), cfg.W, 
        X, cfg.jacobian_cfg, Val(false))
    return Htr
end