include("local_residual.jl")
include("residual.jl")

# A \ ∇
function build_newton_direction!(::Type{Gtype}, d::Vector{T}, A::BlockBandedMatrix{T}, ∇::Vector{T}, cfg=BlockCholeskyStaticConfig(∇, Val(param_size(Gtype)))) where{Gtype<:AbstractWavePacket, T<:Real}
    return block_tridiagonal_cholesky_solver_static!(d, A, ∇, Val(param_size(Gtype)), cfg)
end

#
function schrodinger_gaussian_linesearch(::Type{Gtype}, U::Vector{T}, ∇::Vector{T}, X::Vector{T}, d::Vector{T},
                                        a::T, b::T, Lt::Int,
                                        G0::AbstractVector{<:AbstractWavePacket},
                                        apply_op,
                                        Gf, Gg,
                                        cfg=SchGaussianGradientCFG(Gtype, Lt, U)) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    function ϕ(α)
        @. U = X + α * d
        return  schrodinger_gaussian_residual(Gtype, a, b, Lt, G0, apply_op, Gf, Gg, U)
    end
    function dϕ(α)
        @. U = X + α * d
        schrodinger_gaussian_gradient!(Gtype, ∇, a, b, Lt, G0, apply_op, Gf, Gg, U, cfg)
        return dot(d, ∇)
    end
    function ϕdϕ(α)
        @. U = X + α * d
        val = schrodinger_gaussian_residual(Gtype, a, b, Lt, G0, apply_op, Gf, Gg, U)
        schrodinger_gaussian_gradient!(Gtype, ∇, a, b, Lt, G0, apply_op, Gf, Gg, U, cfg)
        return (val, dot(d, ∇))
    end

    # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
    # More precisely, the real part of the variance cannot be more than halved
    alphamax = typemax(T)
    for k in 1:Lt
        re_z = real.(unpack_gaussian_parameter_z(Gtype, X, (k-1) * psize + 1))
        re_z_dir = real.(unpack_gaussian_parameter_z(Gtype, d, (k-1) * psize + 1))
        w = @. ifelse(re_z_dir < 0, -re_z / (2*re_z_dir), typemax(T))
        alphamax = min(alphamax, minimum(w))
    end
    ls = HagerZhang{T}(;alphamax=alphamax)

    (ϕ_0, dϕ_0) = ϕdϕ(zero(T))
    α_opt, ϕ_0 = ls(ϕ, dϕ, ϕdϕ, min(one(T), alphamax), ϕ_0, dϕ_0)

    return α_opt, ϕ_0
end

mutable struct SchBestGaussianCFG{T, CG, CM, Cchol}
    X::Vector{T}
    U::Vector{T}
    ∇::Vector{T}
    d::Vector{T}
    A::BlockBandedMatrix{T}
    cfg_gradient::CG
    cfg_metric::CM
    cfg_cholesky::Cchol
end
function SchBestGaussianCFG(::Type{Gtype}, ::Type{T}, Lt::Int) where{Gtype<:AbstractWavePacket, T<:Real}
    psize = param_size(Gtype)
    X = zeros(T, psize * Lt)  #Current parameters
    U = similar(X)  #Buffer for sets of parameters
    ∇ = similar(X)  #Buffer for the gradient
    d = similar(X)  #Descent direction
    A = BlockBandedMatrix(Diagonal(zeros(T, Lt * psize)), #Buffer for the metric
        fill(psize, Lt), fill(psize, Lt), (1,1))
    cfg_gradient = SchGaussianGradientCFG(Gtype, Lt, U)
    cfg_metric = SchGaussianGradientAndMetricCFG(Gtype, Lt, X)
    cfg_cholesky = BlockCholeskyStaticConfig(∇, Val(psize))
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
function schrodinger_best_gaussian(::Type{Gtype}, ::Type{T}, a::T, b::T, Lt::Int,
                                        Ginit, apply_op, Gf, Gg,
                                        abs_tol::T,
                                        cfg=SchBestGaussianCFG(Gtype, T, Lt);
                                        maxiter::Int = 1000, verbose::Bool=false) where{Gtype<:AbstractWavePacket, T<:AbstractFloat}
    psize = param_size(Gtype)
    
    #Allocating buffers
    X = cfg.X

    verbose && println("Computing an approximation of the initial condition")
    G_approx_init = gaussian_approx(Gtype, T, Ginit, unpack_gaussian_parameters(Gtype, @SVector rand(T, psize)); verbose=verbose, maxiter=100*maxiter)
    
    # Fills X with the approximation of the initial condition
    for k=1:Lt
        pack_gaussian_parameters!(X, G_approx_init, (k-1) * psize + 1)
    end
    # E0 = schrodinger_gaussian_residual(a, b, Lt, Ginit, apply_op, Gf, Gg, X)
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
    E0 = schrodinger_gaussian_residual(Gtype, a, b, Lt, Ginit, apply_op, Gf, Gg, X)
    while iter < maxiter
        iter += 1
        verbose && println("Iteration $iter on $maxiter :")

        #Natural Gradient descent step
        schrodinger_gaussian_gradient_and_metric!(Gtype, ∇, A, a, b, Lt, Ginit, apply_op, Gf, Gg, X, cfg_metric)
        build_newton_direction!(Gtype, d, A, ∇, cfg_cholesky)
        res = sqrt(max(dot(d, ∇), zero(T)) / 2)
        @. d = T(-0.5) * d
        α, E = schrodinger_gaussian_linesearch(Gtype, U, ∇, X, d, a, b, Lt, Ginit, apply_op, Gf, Gg, cfg_gradient)
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
    G = [unpack_gaussian_parameters(Gtype, X, (k-1)*psize + 1) for k in 1:Lt]
    return G, schrodinger_gaussian_residual(Gtype, a, b, Lt, Ginit, apply_op, Gf, Gg, X)
end