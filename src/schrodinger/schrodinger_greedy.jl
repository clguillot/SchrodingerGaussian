
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
function schrodinger_gaussian_greedy(a::T, b::T, Lt::Int, G0::AbstractVector{<:AbstractWavePacket}, apply_op,
                                        nb_terms::Int;
                                        maxiter::Int = 1000,
                                        verbose::Bool=false, fullverbose::Bool=false) where{T<:AbstractFloat}
    
    verbose = verbose || fullverbose

    n0 = length(G0)

    G0_ = zeros(eltype(G0), n0 + nb_terms)
    G0_[1 : n0] = G0

    TG = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    Tf = typeof(apply_op(zero(T), zero(TG)))
    Gf_ = zeros(Tf, nb_terms, Lt)
    Gf = zeros(TG, 0, Lt)

    Gg_ = zeros(TG, nb_terms, Lt)

    G = zeros(TG, nb_terms, Lt)
    X = zeros(T, gaussian_param_size * Lt, nb_terms)

    abs_tol = sqrt(eps(T)) #T(1e-4)

    cfg = SchBestGaussianCFG(T, Lt)

    #Orthogonal greedy data
    C0 = gaussian_approx_residual_constant_part(G0)
    GramMatrix = zeros(Complex{T}, nb_terms, nb_terms)
    F = zeros(Complex{T}, nb_terms)
    Λ = zeros(Complex{T}, nb_terms) #Coefficients
    res_list = zeros(T, nb_terms)

    for iter=1:nb_terms
        verbose && println("Computing term $iter...")
        G[iter, :], _ = schrodinger_best_gaussian(a, b, Lt, G0_[1 : n0 + iter - 1], apply_op,
                Gf_[1:iter-1, :], Gg_[1:iter-1, :], abs_tol, cfg;
                maxiter=maxiter, verbose=fullverbose)
        
        #Packs the result
        for k=1:Lt
            pack_gaussian_parameters!((@view X[:, iter]), G[iter, k], (k-1) * gaussian_param_size + 1)
        end

        #Computes the residual
        #PDE
        GramMatrix[iter, iter] = @views schrodinger_gaussian_residual_sesquilinear_part(a, b, Lt, apply_op, X[:, iter], X[:, iter])
        for p=1:iter-1
            μ = @views schrodinger_gaussian_residual_sesquilinear_part(a, b, Lt, apply_op, X[:, p], X[:, iter])
            GramMatrix[p, iter] = μ
            GramMatrix[iter, p] = conj(μ)
        end
        F[iter] = @views schrodinger_gaussian_residual_linear_part(a, b, Lt, G0, apply_op, Gf, X[:, iter])

        # Λ[1:iter] = T(-0.5) .* (GramMatrix[1:iter, 1:iter] \ conj.(F[1:iter]))
        Λ[1:iter] = ones(iter)
        res = real(dot(Λ[1:iter], GramMatrix[1:iter, 1:iter], Λ[1:iter]) + dot(F[1:iter], Λ[1:iter]) + C0)
        verbose && println("Residual = $res")
        res_list[iter] = res
        # println("Λ = ", Λ[1:iter])

        # display(GramMatrix)

        #Fills the right member
        for j=1:iter
            for k=1:Lt
                t = a + (k-1)/(Lt-1) * (b - a)
                Gf_[j, k] = Λ[j] * apply_op(t, G[j, k])
                Gg_[j, k] = -1im * Λ[j] * G[j, k]
            end
            G0_[n0 + j] = - Λ[j] * G[j, 1]
        end

        # res0 = gaussian_approx_residual_constant_part(G0_)
        # println("Residual 0 = $res0")

        verbose && println()
    end

    for iter=1:nb_terms
        for k=1:Lt
            G[iter, k] = Λ[iter] * G[iter, k]
        end
    end

    return G, res_list
end