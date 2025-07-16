include("metric.jl")
include("residual.jl")

function gaussian_approx_gradient_and_metric(::Type{Gtype}, Ginit::AbstractWavePacket{D},
                                        X::AbstractVector{T}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)    
    X = SVector{psize}(X)

    #Gradient
    ∇ = gaussian_approx_gradient(Gtype, Ginit, X)

    #Hessian
    A = gaussian_approx_metric_topright_hessian(Gtype, X, X)

    return ∇, A
end

function _gaussian_approx_linesearch(::Type{Gtype}, ::Type{T}, ls, Ginit, X, d) where{Gtype, T}
    function ϕ(α)
        return T(gaussian_approx_residual(unpack_gaussian_parameters(Gtype, X + α * d), Ginit))
    end
    function dϕ(α)
        gaussian_approx_gradient(Gtype, Ginit, X + α * d)
        return T(dot(d, gaussian_approx_gradient(Gtype, Ginit, X + α * d)))
    end
    function ϕdϕ(α)
        return (ϕ(α), dϕ(α))
    end

    fx = ϕ(zero(T))
    dϕ_0 = dϕ(zero(T))
    α, _ = ls(ϕ, dϕ, ϕdϕ, min(T(0.5), ls.alphamax), fx, dϕ_0)
    return T(α)
end

function gaussian_approx(::Type{Gtype}, ::Type{T}, Ginit::AbstractWavePacket{D};
                            rel_tol::T=sqrt(eps(T)), maxiter::Int=1000, verbose::Bool=false) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)
    abs_tol = rel_tol * norm_L2(Ginit)

    # Global optimization with respect to the linear parameter

    G0 = convert(Gtype, truncate_to_gaussian(argmin(g -> norm2_L2(Ginit - g), Ginit)))
    # G0 = argmax(g -> abs(dot_L2(g, Ginit)), Ginit)
    λ = dot_L2(G0, Ginit) / norm2_L2(G0)
    X = SVector{psize}(pack_gaussian_parameters!(zeros(T, psize), λ * G0))

    iter = 0
    while iter < maxiter
        iter += 1

        verbose && println("Iteration $iter :")

        ∇, A = gaussian_approx_gradient_and_metric(Gtype, Ginit, X)
        chA = cholesky(Symmetric(SMatrix{psize, psize}(A)))
        Sd = chA \ ∇
        res = dot(Sd, ∇)
        d = -Sd

        # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
        # More precisely, the real part of the variance cannot be more than halved
        a = real.(unpack_gaussian_parameter_z(Gtype, X))
        a_dir = real.(unpack_gaussian_parameter_z(Gtype, d))
        alphamax = minimum(@. ifelse(a_dir < 0, -a / (2*a_dir), typemax(T)))
        ls = HagerZhang{T}(;alphamax=alphamax)

        α = _gaussian_approx_linesearch(Gtype, T, ls, Ginit, X, d)
        X = T.(X + α * d)

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

    # InteractiveUtils.@code_warntype norm2_L2(Ginit)

    G = unpack_gaussian_parameters(Gtype, X)
    res = norm2_L2(G) + norm2_L2(Ginit) - 2 * real(dot_L2(G, Ginit))
    return G, res
end