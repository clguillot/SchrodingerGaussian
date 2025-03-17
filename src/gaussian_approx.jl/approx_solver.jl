include("metric.jl")
include("residual.jl")

mutable struct GaussianApproxGradientAndMetricCFG{CG, CM}
    cfg_gradient::CG
    cfg_metric::CM
end
function GaussianApproxGradientAndMetricCFG(::Type{Gtype}, X::Vector{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    cfg_gradient = GaussianApproxGradientCFG(Gtype, X)
    cfg_metric = GaussianApproxMetricTRHessCFG(Gtype, X, X)
    return GaussianApproxGradientAndMetricCFG(cfg_gradient, cfg_metric)
end


function gaussian_approx_gradient_and_metric!(::Type{Gtype}, ∇::Vector{T}, A::Matrix{T},
                                        Ginit::AbstractWavePacket,
                                        X::Vector{T},
                                        cfg=GaussianApproxGradientAndMetricCFG(Gtype, X)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    if size(A) != (psize, psize)
        throw(DimensionMismatch("A must be a square matrix of size $(psize)x$(psize) but has size $(size(Htr))"))
    end
    if length(X) != psize
        throw(DimensionMismatch("X must be a vector of size $psize but has size $(length(X))"))
    end
    if length(∇) != psize
        throw(DimensionMismatch("∇ must be a vector of size $psize but has size $(length(∇))"))
    end
    
    #Gradient
    gaussian_approx_gradient!(Gtype, ∇, Ginit, X, cfg.cfg_gradient)

    #Hessian
    gaussian_approx_metric_topright_hessian!(Gtype, A, X, X, cfg.cfg_metric)

    return ∇, A
end

mutable struct GaussianApproxCFG{T<:Real, CG, CFG}
    U::Vector{T}
    X::Vector{T}
    ∇::Vector{T}
    d::Vector{T}
    A::Matrix{T}
    cfg_gradient::CG
    cfg::CFG
end
function GaussianApproxCFG(::Type{Gtype}, ::Type{T}) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    U = zeros(T, psize)
    X = zeros(T, psize)
    ∇ = zeros(T, psize)
    d = zeros(T, psize)
    A = zeros(T, psize, psize)
    cfg_gradient = GaussianApproxGradientCFG(Gtype, U)
    cfg = GaussianApproxGradientAndMetricCFG(Gtype, X)
    return GaussianApproxCFG(U, X, ∇, d, A, cfg_gradient, cfg)
end

function gaussian_approx(::Type{Gtype}, ::Type{T}, Ginit::AbstractWavePacket,
                            cfg=GaussianApproxCFG(Gtype, T);
                            rel_tol::T=sqrt(eps(T)), maxiter::Int=1000, verbose::Bool=false) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    abs_tol = rel_tol * norm_L2(Ginit)

    # Global optimization with respect to the linear parameter

    G0 = argmin(g -> norm2_L2(Ginit - g), Ginit)
    # G0 = argmax(g -> abs(dot_L2(g, Ginit)), Ginit)
    λ = dot_L2(G0, Ginit) / norm2_L2(G0)
    X = pack_gaussian_parameters!(cfg.X, λ * G0)

    U = cfg.U
    d = cfg.d
    iter = 0
    while iter < maxiter
        iter += 1

        verbose && println("Iteration $iter :")

        ∇, A = gaussian_approx_gradient_and_metric!(Gtype, cfg.∇, cfg.A, Ginit, X, cfg.cfg)
        chA = cholesky(Symmetric(SMatrix{psize, psize}(A)))
        Sd = chA \ SVector{psize}(∇)
        res = dot(Sd, ∇)
        @. d = -Sd

        function ϕ(α)
            @. U = X + α * d
            return gaussian_approx_residual(unpack_gaussian_parameters(Gtype, U), Ginit)
        end
        function dϕ(α)
            @. U = X + α * d
            gaussian_approx_gradient!(Gtype, ∇, Ginit, U, cfg.cfg_gradient)
            return dot(cfg.d, cfg.∇)
        end
        function ϕdϕ(α)
            @. U = X + α * d
            val = gaussian_approx_residual(unpack_gaussian_parameters(Gtype, U), Ginit)
            gaussian_approx_gradient!(Gtype, ∇, Ginit, U, cfg.cfg_gradient)
            return (val, dot(d, ∇))
        end

        # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
        # More precisely, the real part of the variance cannot be more than halved
        a = real.(unpack_gaussian_parameter_z(Gtype, X))
        a_dir = real.(unpack_gaussian_parameter_z(Gtype, d))
        alphamax = minimum(@. ifelse(a_dir < 0, -a / (2*a_dir), typemax(T)))
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

    G = unpack_gaussian_parameters(Gtype, X)
    res = norm2_L2(G) + norm2_L2(Ginit) - 2 * real(dot_L2(G, Ginit))
    return G, res
end