
#=
    Provides an approximation of
        i∂ₜu = H(t)u + f(t) + g(t) on (a, b)
        u(a) = ∑ᵣG0[r]
    as a sum of gaussians by minimizing the residual
        |u(a) - ∑ᵣG0[r]|^2 + ∫_(a,b) dt |i∂ₜu(t) - H(t)u(t) - f(t) - g(t)|²
    with a greedy algorithm, where
    - H(t)g = apply_op(t, g)
    - f(t) = ∑ₖ,ᵣ Gf[r, k] ζₖ(t)
    - g(t) = ∑ₖ,ᵣ Gg[r, k] ζₖ'(t)
    Return G::Vector{<:GaussianWavePacket1D}
=#
function schrodinger_gaussian_greedy(::Type{Gtype}, ::Type{T}, a::T, b::T, Lt::Int,
                Ginit::AbstractWavePacket, apply_op, nb_terms::Int;
                maxiter::Int = 1000, verbose::Bool=false, fullverbose::Bool=false, greedy_orthogonal::Bool=true) where{Gtype<:AbstractWavePacket, T<:AbstractFloat}
    
    verbose = verbose || fullverbose
    psize = param_size(Gtype)

    G0_ = zeros(Gtype, nb_terms)

    Tf = typeof(apply_op(a, zero(Gtype)))
    Gf_ = fill(apply_op(a, zero(Gtype)), nb_terms, Lt)
    Gf = zeros(Gtype, 0, Lt)

    Gg_ = zeros(Gtype, nb_terms, Lt)
    Gg = zeros(Gtype, 0, Lt)

    G = zeros(Gtype, nb_terms, Lt)
    X = zeros(T, psize * Lt, nb_terms)

    abs_tol = sqrt(eps(T)) #T(1e-4)

    cfg = SchBestGaussianCFG(Gtype, T, Lt)

    #Orthogonal greedy data
    C0 = gaussian_approx_residual_constant_part(Ginit)
    GramMatrix = zeros(Complex{T}, nb_terms, nb_terms)
    F = zeros(Complex{T}, nb_terms)
    Λ = zeros(Complex{T}, nb_terms) #Coefficients
    res_list = zeros(T, nb_terms)

    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        for iter=1:nb_terms
            verbose && println("Computing term $iter...")
            G[iter, :], _ = @views schrodinger_best_gaussian(Gtype, T, a, b, Lt, Ginit + WavePacketSum(G0_[1 : iter-1]), apply_op,
                    Gf_[1:iter-1, :], Gg_[1:iter-1, :], abs_tol, cfg;
                    maxiter=maxiter, verbose=fullverbose)
            
            #Packs the result
            for k=1:Lt
                pack_gaussian_parameters!((@view X[:, iter]), G[iter, k], (k-1) * psize + 1)
            end

            #Computes the residual
            #PDE
            GramMatrix[iter, iter] = @views schrodinger_gaussian_residual_sesquilinear_part(Gtype, a, b, Lt, apply_op, X[:, iter], X[:, iter])
            for p=1:iter-1
                μ = @views schrodinger_gaussian_residual_sesquilinear_part(Gtype, a, b, Lt, apply_op, X[:, p], X[:, iter])
                GramMatrix[p, iter] = μ
                GramMatrix[iter, p] = conj(μ)
            end
            F[iter] = conj(@views schrodinger_gaussian_residual_linear_part(Gtype, a, b, Lt, Ginit, apply_op, Gf, Gg, X[:, iter]))

            if greedy_orthogonal
                Λ[1:iter] = GramMatrix[1:iter, 1:iter] \ F[1:iter]
            else
                Λ[1:iter] = ones(T, iter)
            end
            res = real(dot(Λ[1:iter], GramMatrix[1:iter, 1:iter], Λ[1:iter]) - 2 * dot(F[1:iter], Λ[1:iter]) + C0)
            verbose && println("Residual = $res")
            res_list[iter] = res
            # println("Λ = ", Λ[1:iter])

            # display(GramMatrix)

            #Fills the right member
            for j=1:iter
                for k=1:Lt
                    g = -Λ[j] * @views unpack_gaussian_parameters(Gtype, X[:, j], (k-1)*psize + 1)
                    t = a + (k-1)/(Lt-1) * (b - a)
                    Gf_[j, k] = apply_op(t, g)
                    Gg_[j, k] = g
                end
                g = -Λ[j] * @views unpack_gaussian_parameters(Gtype, X[:, j])
                G0_[j] = g
            end

            # res0 = gaussian_approx_residual_constant_part(G0_)
            # println("Residual 0 = $res0")

            verbose && println()
        end

        for iter=1:nb_terms
            for k=1:Lt
                G[iter, k] = Λ[iter] * @views unpack_gaussian_parameters(Gtype, X[:, iter], (k-1)*psize + 1)
            end
        end
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end

    return G, res_list
end

#=

=#
function schrodinger_gaussian_greedy_timestep(::Type{Gtype}, ::Type{T}, a::T, b::T, Lt::Int, nb_timesteps::Int,
    Ginit::AbstractWavePacket, apply_op, nb_greedy_terms::Int;
    progressbar::Bool=false, maxiter::Int = 1000, verbose::Bool=false, fullverbose::Bool=false) where{Gtype<:AbstractWavePacket, T<:AbstractFloat}

    res = zero(T)
    G = zeros(Gtype, nb_greedy_terms, Lt)
    lt = fld(Lt, nb_timesteps)
    h = (b-a) / (Lt-1)
    for p in (progressbar ? ProgressBar(1:nb_timesteps) : 1:nb_timesteps)
        k1 = (p-1)*lt + 1
        k2 = (p == nb_timesteps) ? Lt : p*lt + 1
        a_ = a + (k1-1)*h
        b_ = a + (k2-1)*h
        lt_ = k2 - k1 + 1
        G0_ = (p == 1) ? Ginit : WavePacketSum(@view G[:, k1])
        G_block, res_list = schrodinger_gaussian_greedy(Gtype, T, a_, b_, lt_, G0_, apply_op, nb_greedy_terms; greedy_orthogonal=false, maxiter=maxiter, verbose=verbose, fullverbose=fullverbose)
        @views G[:, k1:k2] .= G_block
        res += sqrt(res_list[end])
    end

    return G, res
end