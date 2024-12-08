
#=

    METRIC

=#

#=
    Computes
        ∫_(0,h) ds <i∂ₜ(ζ₀(s)F0 + ζ₁(s)F1), i∂ₜ(ζ₀(s)G0 + ζ₁(s)G1)>
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - F0, F1 are obtained by unpacking respectively X1[1:6] and X1[7:12]
        - G0, G1 are obtained by unpacking respectively X2[1:6] and X2[7:12]
=#
@fastmath function schrodinger_gaussian_metric(X1::AbstractVector{T1}, X2::AbstractVector{T2}, ::Val{check_len}=Val(true)) where{T1<:Real, T2<:Real, check_len}
    
    if check_len && (length(X1) != 2*gaussian_param_size || length(X2) != 2*gaussian_param_size)
        throw(DimensionMismatch("The size of X1 and X2 must be equal to $(2*gaussian_param_size)"))
    end
    
    F0 = unpack_gaussian_parameters(X1, 1)
    F1 = unpack_gaussian_parameters(X1, gaussian_param_size + 1)

    G0 = unpack_gaussian_parameters(X2, 1)
    G1 = unpack_gaussian_parameters(X2, gaussian_param_size + 1)

    # Sesquilinear part

    # <i∂ₜ,i∂ₜ>
    S = real(dot_L2(F0, G0)) + real(dot_L2(F1, G1))
    S -= real(dot_L2(F0, G1)) + real(dot_L2(F1, G0))

    return S
end

mutable struct SchGaussianMetricTRHessCFG{T}
    W::Vector{T}
    gradient_cfg::ForwardDiff.GradientConfig
    jacobian_cfg::ForwardDiff.JacobianConfig
end

function SchGaussianMetricTRHessCFG(X::Vector{T}) where{T<:Real}
    if length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end

    W = zeros(T, 2*gaussian_param_size)
    jacobian_cfg = ForwardDiff.JacobianConfig(x -> nothing, W, X, ForwardDiff.Chunk(2*gaussian_param_size))
    gradient_cfg = ForwardDiff.GradientConfig(jacobian_cfg, X, ForwardDiff.Chunk(2*gaussian_param_size))
    return SchGaussianMetricTRHessCFG(W, gradient_cfg, jacobian_cfg)
end

#=
    Computes (∂ₓ₁∂ₓ₂E)(X, X) where E(x1, x2) = schrodinger_gaussian_metric(x1, x2)
=#
function schrodinger_gaussian_metric_topright_hessian!(Htr::Matrix{T}, h::T, X::Vector{T},
                                                cfg::SchGaussianMetricTRHessCFG=SchGaussianMetricTRHessCFG(X)) where{T<:Real}
    if !all(n -> n == 2*gaussian_param_size, size(Htr))
        throw(DimensionMismatch("Htr must be a square matrix of size ($(2*gaussian_param_size), $(2*gaussian_param_size)) but has size $(size(Htr))"))
    end
    if length(X) != 2*gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $(2*gaussian_param_size) but has size $(length(X))"))
    end

    f(Y1, Y2) = schrodinger_gaussian_metric(Y1, Y2, Val(false))
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1, cfg.gradient_cfg, Val(false))
    ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X, Z), cfg.W, 
        X, cfg.jacobian_cfg, Val(false))
    Htr ./= h
    return Htr
end


#=

    TIME STEP METRIC

=#


#=
    Computes the local residual
        ∫_(0,h) ds <i∂ₜ(ζ₁(s)F1), i∂ₜ(ζ₁(s)G1)>
    where
        - ζ₀, ζ₁ are the 2 P1 finite element functions on (0,h)
        - F0, F1 are obtained by unpacking respectively X1[1:6] and Y1[1:6]
        - G0, G1 are obtained by unpacking respectively X2[1:6] and Y2[1:6]
=#
function schrodinger_gaussian_metric_time_step(h::T, X1::AbstractVector{T1}, X2::AbstractVector{T2},
                                            ::Val{check_len}=Val(true)) where{T<:Real, T1<:Real, T2<:Real, check_len}
    
    if check_len && (length(X1) != gaussian_param_size || length(X2) != gaussian_param_size)
        throw(DimensionMismatch("The size of X1 and X2 must be equal to $(gaussian_param_size)"))
    end
    
    F1 = unpack_gaussian_parameters(X1)
    G1 = unpack_gaussian_parameters(X2)

    #|i∂ₜ|^2
    return real(dot_L2(F1, G1)) / h
end

#=
    Computes (∂ₓ₁∂ₓ₂E)(X, X) where E(x1, x2) = schrodinger_gaussian_metric_time_step(x1, x2)
=#
function schrodinger_gaussian_metric_time_step_topright_hessian!(Htr::DenseMatrix{T}, h::T, X::AbstractVector{T}) where{T<:Real}
    if !allequal([size(Htr)..., gaussian_param_size])
        throw(DimensionMismatch("Htr must be a square matrix of size $(gaussian_param_size)x$(gaussian_param_size)"))
    end
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of length $gaussian_param_size but has length $(length(X))"))
    end

    f(Y1, Y2) = schrodinger_gaussian_metric_time_step(h, Y1, Y2, Val(false))
    ∇₁f!(∇, Y1, Y2) = ForwardDiff.gradient!(∇, Z -> f(Z, Y2), Y1)
    W = zeros(T, gaussian_param_size)
    return ForwardDiff.jacobian!(Htr, (∇, Z) -> ∇₁f!(∇, X, Z), W, X)
end