include("local_residual.jl")
include("residual.jl")

# A \ ∇
function build_newton_direction!(::Type{Gtype}, d::Vector{T}, A::BlockBandedMatrix{T}, ∇::Vector{T}, cfg=BlockCholeskyStaticConfig(∇, Val(param_size(Gtype)))) where{Gtype<:AbstractWavePacket, T<:Real}
    return block_tridiagonal_cholesky_solver_static!(d, A, ∇, Val(param_size(Gtype)), cfg)
end

#
function schrodinger_gaussian_linesearch(::Type{Gtype}, U::Vector{T}, ∇::Vector{T}, X::Vector{T}, d::Vector{T},
                                        a::T, b::T, Lt::Int,
                                        G0::AbstractWavePacket, apply_op,
                                        Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket},
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
        val, _ = schrodinger_gaussian_gradient!(Gtype, ∇, a, b, Lt, G0, apply_op, Gf, Gg, U, cfg)
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
        u(a) = Ginit
    by minimizing the residual
        |u(a) - Ginit|^2 + ∫_(a,b) dt |i∂ₜu(t) - H(t)u(t) - f(t) - g(t)|²
    where
    - u = ∑ₖ G[k] ζₖ(t)
    - H(t)g = apply_op(t, g)
    - f(t) = ∑ₖ,ᵣ Gf[r, k] ζₖ(t)
    - g(t) = ∑ₖ,ᵣ Gg[r, k] ζₖ'(t)
    Return G::Vector{<:GaussianWavePacket1D}
=#
function schrodinger_best_gaussian(::Type{Gtype}, ::Type{T}, a::T, b::T, Lt::Int,
                                        Ginit::AbstractWavePacket, apply_op,
                                        Gf::AbstractMatrix{<:AbstractWavePacket}, Gg::AbstractMatrix{<:AbstractWavePacket},
                                        abs_tol::T,
                                        cfg=SchBestGaussianCFG(Gtype, T, Lt);
                                        maxiter::Int = 1000, verbose::Bool=false) where{Gtype<:AbstractWavePacket, T<:AbstractFloat}
    psize = param_size(Gtype)

    h = (b-a)/(Lt-1)
    
    #Allocating buffers
    X = cfg.X

    # Approximation of the initial condition
    # Starts an approximation process from every center of Ginit, and takes the best
    verbose && println("Computing an approximation of the initial condition")
    G_approx_init, _ = gaussian_approx(Gtype, T, Ginit; verbose=verbose, maxiter=100*maxiter)
    
    # Minimizer over the basis of the approximation of the initial condition
    # Γ = ones(T, Lt)
    Gram = Tridiagonal(zeros(Complex{T}, Lt-1), zeros(Complex{T}, Lt), zeros(Complex{T}, Lt-1))
    F = zeros(Complex{T}, Lt)
    # PDE residual
    for k in 1:Lt
        t = a + (k-1)*h
        G = G_approx_init
        HG = apply_op(t, G)

        # Linear vector
        for l in max(1,k-1):min(Lt,k+1)
            F[k] += @views schrodinger_gaussian_cross_residual(h, Lt, k, l, G, WavePacketSum(Gg[:, l]), HG, WavePacketSum(Gf[:, l]))
        end

        # Gram matrix
        Gram[k, k] = schrodinger_gaussian_square_residual(h, Lt, k, G, HG)
        if k < Lt
            Gram[k, k+1] = schrodinger_gaussian_cross_residual(h, Lt, k, k+1, G, G, HG, apply_op(t+h, G))
            Gram[k+1, k] = conj(Gram[k, k+1])
        end
    end
    # Initial condition
    F .*= (b - a)
    F[1] += dot_L2(G_approx_init, Ginit)
    Gram .*= (b - a)
    Gram[1, 1] += dot_L2(G_approx_init, G_approx_init)
    Γ = Gram' \ F

    # Fills X with the approximation of the initial condition
    for k=1:Lt
        G = Γ[k] * G_approx_init
        pack_gaussian_parameters!(X, G, (k-1) * psize + 1)
    end

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