include("local_metric.jl")
include("local_residual.jl")
include("residual.jl")

#=
    ...
=#
function schrodinger_best_gaussian_locally_optimized(a::T, b::T, Lt::Int, G0::GaussianWavePacket1D,
                                                        apply_op, Gop::Gtype,
                                                        Gf::Matrix{<:GaussianWavePacket1D},
                                                        Gg::Matrix{<:GaussianWavePacket1D},
                                                        maxiter::Int = 20) where{T<:AbstractFloat, Gtype}
    h = (b - a) / (Lt - 1)
    X = zeros(T, gaussian_param_size * Lt)
    pack_gaussian_parameters!(X, G0)

    Yk = zeros(T, gaussian_param_size)  #Previous parameters
    Y = zeros(T, gaussian_param_size)   #Current parameters
    U = zeros(T, gaussian_param_size)   #Buffer for parameter set
    ∇ = zeros(T, gaussian_param_size)   #Buffer for gradient
    diff_res = DiffResults.GradientResult(Y)    #Buffer for gradient and value
    H = zeros(T, gaussian_param_size, gaussian_param_size)  #Buffer for hessian

    for k=1:Lt-1
        t = a + (k-1) * h
        Yk .= X[(k-1)*gaussian_param_size + 1 : k*gaussian_param_size]
        Gk = unpack_gaussian_parameters(Yk, 1)
        ε = sqrt(real(dot_L2(Gk, Gk))) * (eps(T) / h)^(3/4)

        Gfk = @view Gf[:, k]
        Gfkp1 = @view Gf[:, k+1]
        Ggk = @view Gg[:, k]
        Ggkp1 = @view Gg[:, k+1]
        f(Y) = schrodinger_gaussian_local_residual(t, h, apply_op, Gop, Gfk, Gfkp1, Ggk, Ggkp1, [Yk ; Y])
        function fg!(∇, Y)
            ForwardDiff.gradient!(∇, f, Y)
        end

        iter = 0
        converged = false
        fx = f(Yk)
        Y .= Yk
        res = typemax(T)
        while iter < maxiter
            fg!(∇, Y)
            schrodinger_gaussian_metric_time_step_topright_hessian!(H, h, Y)
            d = - H \ ∇
            res = sqrt(abs(dot(d, ∇)))

            if res <= ε
                converged = true
                break
            end

            function ϕ(α)
                @. U = Y + α * d
                return f(U)
            end
            function dϕ(α)
                @. U = Y + α * d
                fg!(diff_res, U)
                return dot(d, DiffResults.gradient(diff_res))
            end
            function ϕdϕ(α)
                @. U = Y + α * d
                fg!(diff_res, U)
                return (DiffResults.value(diff_res), dot(d, DiffResults.gradient(diff_res)))
            end

            # Creates a linesearch with an alphamax to avoid negative variance for the gaussians
            # More precisely, the real part of the variance cannot be more than halved
            rez = real(unpack_gaussian_parameter_z(Y))
            rez_dir = real(unpack_gaussian_parameter_z(d))
            alphamax = (rez_dir < 0) ? -rez / (2*rez_dir) : typemax(T)
            ls = HagerZhang{T}(;alphamax=alphamax)

            dϕ_0 = dϕ(zero(T))
            α, fx = ls(ϕ, dϕ, ϕdϕ, min(T(0.5), alphamax), fx, dϕ_0)
            @. Y += α * d

            iter += 1
        end
        @views X[k*gaussian_param_size + 1 : (k+1)*gaussian_param_size] .= Y
    end

    G = zeros(GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}, Lt)
    for k=1:Lt
        G[k] = unpack_gaussian_parameters(X, (k-1)*gaussian_param_size + 1)
    end

    return G
end

# A \ ∇
function build_newton_direction!(d::Vector{T}, A::BlockBandedMatrix{T}, ∇::Vector{T}, cfg::BlockCholeskyStaticConfig=BlockCholeskyStaticConfig(∇, Val(gaussian_param_size))) where{T <: AbstractFloat}
    return block_tridiagonal_cholesky_solver_static!(d, A, ∇, Val(gaussian_param_size), cfg)
end

#
function schrodinger_gaussian_linesearch(U::Vector{T}, ∇::Vector{T}, X::Vector{T}, d::Vector{T},
                                        a::T, b::T, Lt::Int,
                                        G0::AbstractVector{<:GaussianWavePacket1D},
                                        apply_op, Gop::Gtype,
                                        Gf::AbstractMatrix{<:GaussianWavePacket1D},
                                        Gg::AbstractMatrix{<:GaussianWavePacket1D},
                                        cfg::SchGaussianGradientCFG=SchGaussianGradientCFG(U)) where{T<:Real, Gtype}
    function ϕ(α)
        @. U = X + α * d
        return  schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, U)
    end
    function dϕ(α)
        @. U = X + α * d
        schrodinger_gaussian_gradient!(∇, a, b, Lt, G0, apply_op, Gop, Gf, Gg, U, cfg)
        return dot(d, ∇)
    end
    function ϕdϕ(α)
        @. U = X + α * d
        val = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, U)
        schrodinger_gaussian_gradient!(∇, a, b, Lt, G0, apply_op, Gop, Gf, Gg, U, cfg)
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

struct SchBestGaussianCFG{T}
    X::Vector{T}
    U::Vector{T}
    ∇::Vector{T}
    d::Vector{T}
    A::BlockBandedMatrix{T}
    cfg_gradient::SchGaussianGradientCFG{T}
    cfg_metric::SchGaussianGradientAndMetricCFG{T}
    cfg_cholesky::BlockCholeskyStaticConfig{T}
end
function SchBestGaussianCFG(::Type{T}, Lt::Int) where{T<:Real}
    X = zeros(T, gaussian_param_size * Lt)  #Current parameters
    U = similar(X)  #Buffer for sets of parameters
    ∇ = similar(X)  #Buffer for the gradient
    d = similar(X)  #Descent direction
    A = BlockBandedMatrix(Diagonal(zeros(T, Lt * gaussian_param_size)), #Buffer for the metric
        fill(gaussian_param_size, Lt), fill(gaussian_param_size, Lt), (1,1))
    cfg_gradient = SchGaussianGradientCFG(U)
    cfg_metric = SchGaussianGradientAndMetricCFG(T)
    cfg_cholesky = BlockCholeskyStaticConfig(∇, Val(gaussian_param_size))
    return SchBestGaussianCFG(X, U, ∇, d, A, cfg_gradient, cfg_metric, cfg_cholesky)
end

#=
    Provides the best gaussian approximation to
        i∂ₜu = (-Δ + v)u + f + g on (a, b)
        u(a) = G0
    by minimizing the residual
        ∫_(a,b) dt |i∂ₜu(t) - (-Δ + v)u(t) - f(t) - g(t)|²
    where
    - u = ∑ₖ G[k] ζₖ(t)
    - v = Gv
    - f(t) = ∑ₖ,ᵣ Gf[r, k] ζₖ(t)
    - g(t) = ∑ₖ,ᵣ Gg[r, k] ζₖ'(t)
    Return G::Vector{Gaussian{T}}
=#
function schrodinger_best_gaussian(a::T, b::T, Lt::Int, G0::AbstractVector{<:GaussianWavePacket1D},
                                        apply_op, Gop::Gtype,
                                        Gf::AbstractMatrix{<:GaussianWavePacket1D},
                                        Gg::AbstractMatrix{<:GaussianWavePacket1D},
                                        abs_tol::T,
                                        cfg::SchBestGaussianCFG=SchBestGaussianCFG(T, Lt);
                                        maxiter::Int = 1000,
                                        verbose::Bool=false) where{T<:AbstractFloat, Gtype}
    
    
    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    
    #Allocating buffers
    X = cfg.X

    verbose && println("Computing an approximation of the initial condition")
    @time Ginit = gaussian_approx(G0, unpack_gaussian_parameters(rand(T, gaussian_param_size)); verbose=verbose)
    
    # Fills X with the approximation of the initial condition
    # for k=1:Lt
    #     pack_gaussian_parameters!(X, Ginit, (k-1) * gaussian_param_size + 1)
    # end
    # E0 = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, X)
    # println("Initial condition residual = $E0")

    verbose && println("Computing a time stepping approximation")
    pack_gaussian_parameters!(X, Ginit, 1)
    @time G_loc_opt = schrodinger_best_gaussian_locally_optimized(a, b, Lt, Ginit, apply_op, Gop, Gf, Gg)
    for k=2:Lt
        pack_gaussian_parameters!(X, G_loc_opt[k], (k-1)*gaussian_param_size + 1)
    end
    verbose && println("Locally optimized Residual = ", schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, X))

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
    E0 = schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, X)
    @time while iter < maxiter
        iter += 1
        verbose && println("Iteration $iter on $maxiter :")

        #Natural Gradient descent step
        schrodinger_gaussian_gradient_and_metric!(∇, A, a, b, Lt, G0, apply_op, Gop, Gf, Gg, X, cfg_metric)
        build_newton_direction!(d, A, ∇, cfg_cholesky)
        res = sqrt(max(dot(d, ∇), zero(T)) / 2)
        @. d = T(-0.5) * d
        α, E = schrodinger_gaussian_linesearch(U, ∇, X, d, a, b, Lt, G0, apply_op, Gop, Gf, Gg, cfg_gradient)
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

    println("Number of iterations : $iter")

    #Unpacking the result
    G = Vector{GT}(undef, Lt)
    for k=1:Lt
        G[k] = unpack_gaussian_parameters(X, (k-1)*gaussian_param_size + 1)
    end

    return G, schrodinger_gaussian_residual(a, b, Lt, G0, apply_op, Gop, Gf, Gg, X)
end