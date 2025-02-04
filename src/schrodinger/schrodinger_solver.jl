include("local_residual.jl")
include("residual.jl")

# A \ ∇
function build_newton_direction!(d::Vector{T}, A::BlockBandedMatrix{T}, ∇::Vector{T}, cfg::BlockCholeskyStaticConfig=BlockCholeskyStaticConfig(∇, Val(gaussian_param_size))) where{T <: AbstractFloat}
    return block_tridiagonal_cholesky_solver_static!(d, A, ∇, Val(gaussian_param_size), cfg)
end

#
function schrodinger_gaussian_linesearch(U::Vector{T}, ∇::Vector{T}, X::Vector{T}, d::Vector{T},
                                        a::T, b::T, Lt::Int,
                                        G0::AbstractVector{<:AbstractWavePacket},
                                        apply_op,
                                        Gf, Gg,
                                        cfg::SchGaussianGradientCFG=SchGaussianGradientCFG(Lt, U)) where{T<:Real}
    function ϕ(α)
        @. U = X + α * d
        return  schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gf, Gg, U)
    end
    function dϕ(α)
        @. U = X + α * d
        schrodinger_gaussian_gradient!(∇, a, b, Lt, G0, apply_op, Gf, Gg, U, cfg)
        return dot(d, ∇)
    end
    function ϕdϕ(α)
        @. U = X + α * d
        val = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gf, Gg, U)
        schrodinger_gaussian_gradient!(∇, a, b, Lt, G0, apply_op, Gf, Gg, U, cfg)
        return (val, dot(d, ∇))
    end

    # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
    # More precisely, the real part of the variance cannot be more than halved
    alphamax = typemax(T)
    for k in 1:Lt
        rez = real(unpack_gaussian_parameter_z(X, (k-1) * gaussian_param_size + 1))
        rez_dir = real(unpack_gaussian_parameter_z(d, (k-1) * gaussian_param_size + 1))
        if rez_dir < 0
            alphamax = min(alphamax, -rez / (2*rez_dir))
        end
    end
    ls = HagerZhang{T}(;alphamax=alphamax)

    (ϕ_0, dϕ_0) = ϕdϕ(zero(T))
    α_opt, ϕ_0 = ls(ϕ, dϕ, ϕdϕ, min(one(T), alphamax), ϕ_0, dϕ_0)

    return α_opt, ϕ_0
end

struct SchBestGaussianCFG{T, CG, CM, Cchol}
    X::Vector{T}
    U::Vector{T}
    ∇::Vector{T}
    d::Vector{T}
    A::BlockBandedMatrix{T}
    cfg_gradient::CG
    cfg_metric::CM
    cfg_cholesky::Cchol
end
function SchBestGaussianCFG(::Type{T}, Lt::Int) where{T<:Real}
    X = zeros(T, gaussian_param_size * Lt)  #Current parameters
    U = similar(X)  #Buffer for sets of parameters
    ∇ = similar(X)  #Buffer for the gradient
    d = similar(X)  #Descent direction
    A = BlockBandedMatrix(Diagonal(zeros(T, Lt * gaussian_param_size)), #Buffer for the metric
        fill(gaussian_param_size, Lt), fill(gaussian_param_size, Lt), (1,1))
    cfg_gradient = SchGaussianGradientCFG(Lt, U)
    cfg_metric = SchGaussianGradientAndMetricCFG(Lt, X)
    cfg_cholesky = BlockCholeskyStaticConfig(∇, Val(gaussian_param_size))
    return SchBestGaussianCFG(X, U, ∇, d, A, cfg_gradient, cfg_metric, cfg_cholesky)
end

#=
    Provides the best gaussian approximation to
        i∂ₜu = H(t)u + f(t) + g(t) on (a, b)
        u(a) = ∑ᵣG0[r]
    by minimizing the residual
        |u(a) - ∑ᵣG0[r]|^2 + ∫_(a,b) dt |i∂ₜu(t) - H(t)u(t) - f(t) - g(t)|²
    where
    - u = ∑ₖ G[k] ζₖ(t)
    - H(t)g = apply_op(t, g)
    - f(t) = ∑ₖ,ᵣ Gf[r, k] ζₖ(t)
    - g(t) = ∑ₖ,ᵣ Gg[r, k] ζₖ'(t)
    Return G::Vector{<:GaussianWavePacket1D}
=#
function schrodinger_best_gaussian(a::T, b::T, Lt::Int, G0::AbstractVector{<:AbstractWavePacket1D},
                                        apply_op,
                                        Gf, Gg,
                                        abs_tol::T,
                                        cfg=SchBestGaussianCFG(T, Lt);
                                        maxiter::Int = 1000,
                                        verbose::Bool=false) where{T<:AbstractFloat}
    
    
    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    
    #Allocating buffers
    X = cfg.X

    verbose && println("Computing an approximation of the initial condition")
    Ginit = gaussian_approx(G0, unpack_gaussian_parameters(rand(T, gaussian_param_size)); verbose=verbose, maxiter=100*maxiter)
    
    # Fills X with the approximation of the initial condition
    for k=1:Lt
        pack_gaussian_parameters!(X, Ginit, (k-1) * gaussian_param_size + 1)
    end
    # E0 = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gf, Gg, X)
    # println("Initial condition residual = $E0")

    #Gradient and Hessian
    U = cfg.U
    ∇ = cfg.∇
    d = cfg.d
    A = cfg.A
    cfg_gradient = cfg.cfg_gradient
    cfg_metric = cfg.cfg_metric
    cfg_cholesky = cfg.cfg_cholesky

    # Global space-time iterations
    iter = 0
    E0 = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gf, Gg, X)
    while iter < maxiter
        iter += 1
        verbose && println("Iteration $iter on $maxiter :")

        #Natural Gradient descent step
        schrodinger_gaussian_gradient_and_metric!(∇, A, a, b, Lt, G0, apply_op, Gf, Gg, X, cfg_metric)
        build_newton_direction!(d, A, ∇, cfg_cholesky)
        res = sqrt(max(dot(d, ∇), zero(T)) / 2)
        @. d = T(-0.5) * d
        α, E = schrodinger_gaussian_linesearch(U, ∇, X, d, a, b, Lt, G0, apply_op, Gf, Gg, cfg_gradient)
        @. X += α * d
        verbose && println("res = $res")

        if verbose
            println("E = $E")
            println("α = $α")
            println()
        end
        if res < abs_tol
            break
        end
    end

    verbose && println("Number of iterations : $iter")

    #Unpacking the result
    G = Vector{GT}(undef, Lt)
    for k=1:Lt
        G[k] = unpack_gaussian_parameters(X, (k-1)*gaussian_param_size + 1)
    end

    return G, schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gf, Gg, X)
end