include("metric.jl")
include("residual.jl")

mutable struct GaussianApproxGradientAndMetricCFG
    cfg_gradient::GaussianApproxGradientCFG
    cfg_metric::GaussianApproxMetricTRHessCFG
end
function GaussianApproxGradientAndMetricCFG(X::Vector{T}) where{T<:Real}
    cfg_gradient = GaussianApproxGradientCFG(X)
    cfg_metric = GaussianApproxMetricTRHessCFG(X)
    return GaussianApproxGradientAndMetricCFG(cfg_gradient, cfg_metric)
end


function gaussian_approx_gradient_and_metric!(∇::Vector{T}, A::Matrix{T},
                                        G_list::AbstractVector{<:GaussianWavePacket1D},
                                        X::Vector{T},
                                        cfg::GaussianApproxGradientAndMetricCFG=GaussianApproxGradientAndMetricCFG(X)) where{T<:Real}
    if size(A) != (gaussian_param_size, gaussian_param_size)
        throw(DimensionMismatch("A must be a square matrix of size $(gaussian_param_size)x$(gaussian_param_size) but has size $(size(Htr))"))
    end
    if length(X) != gaussian_param_size
        throw(DimensionMismatch("X must be a vector of size $gaussian_param_size but has size $(length(X))"))
    end
    
    #Gradient
    gaussian_approx_gradient!(∇, G_list, X, cfg.cfg_gradient)

    #Hessian
    gaussian_approx_metric_topright_hessian!(A, X, X, cfg.cfg_metric)

    return ∇, A
end

mutable struct GaussianApproxCFG{T<:Real}
    U::Vector{T}
    X::Vector{T}
    ∇::Vector{T}
    d::Vector{T}
    A::Matrix{T}
    cfg_gradient::GaussianApproxGradientCFG
    cfg::GaussianApproxGradientAndMetricCFG
end
function GaussianApproxCFG(::Type{T}, G_list::AbstractVector{<:GaussianWavePacket1D}) where{T<:Real}
    U = zeros(T, gaussian_param_size)
    X = zeros(T, gaussian_param_size)
    ∇ = zeros(T, gaussian_param_size)
    d = zeros(T, gaussian_param_size)
    A = zeros(T, gaussian_param_size, gaussian_param_size)
    cfg_gradient = GaussianApproxGradientCFG(U)
    cfg = GaussianApproxGradientAndMetricCFG(X)
    return GaussianApproxCFG(U, X, ∇, d, A, cfg_gradient, cfg)
end

function gaussian_approx(G_list::AbstractVector{<:GaussianWavePacket1D},
                            G_initial_guess::GaussianWavePacket1D{Complex{T}, Complex{T}, T, T},
                            cfg::GaussianApproxCFG=GaussianApproxCFG(T, G_list);
                            rel_tol::T=sqrt(eps(T)), maxiter::Int=1000, verbose::Bool=false) where{T<:Real}
    abs_tol = rel_tol * gaussian_approx_residual_constant_part(G_list)
    X = pack_gaussian_parameters!(cfg.X, G_initial_guess)
    U = cfg.U
    d = cfg.d
    iter = 0
    while iter < maxiter
        iter += 1

        verbose && println("Iteration $iter :")

        ∇, A = gaussian_approx_gradient_and_metric!(cfg.∇, cfg.A, G_list, X, cfg.cfg)
        chA = cholesky(Symmetric(SMatrix{gaussian_param_size, gaussian_param_size}(A)))
        Sd = chA \ SVector{gaussian_param_size}(∇)
        res = dot(Sd, ∇)
        @. d = -Sd

        function ϕ(α)
            @. U = X + α * d
            return gaussian_approx_residual(U, G_list)
        end
        function dϕ(α)
            @. U = X + α * d
            gaussian_approx_gradient!(∇, G_list, U, cfg.cfg_gradient)
            return dot(cfg.d, cfg.∇)
        end
        function ϕdϕ(α)
            @. U = X + α * d
            val = gaussian_approx_residual(U, G_list)
            gaussian_approx_gradient!(∇, G_list, U, cfg.cfg_gradient)
            return (val, dot(d, ∇))
        end

        # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
        # More precisely, the real part of the variance cannot be more than halved
        a = real(unpack_gaussian_parameter_z(X))
        a_dir = real(unpack_gaussian_parameter_z(d))
        alphamax = (a_dir < 0) ? -a / (2*a_dir) : typemax(T)
        ls = HagerZhang{T}(;alphamax=alphamax)

        fx = ϕ(zero(T))
        dϕ_0 = dϕ(zero(T))
        α, fx = ls(ϕ, dϕ, ϕdϕ, min(T(0.5), alphamax), fx, dϕ_0)
        @. X += α * d

        if verbose
            println("α = $α")
            println("res = $res")
            println()
        end

        if res < abs_tol
            break
        end
    end

    if verbose
        if iter < maxiter
            println("Converged after $iter iterations")
        else
            println("Not converged after $iter iterations")
        end
    end

    return unpack_gaussian_parameters(X)
end