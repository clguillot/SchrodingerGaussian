
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
function schrodinger_gaussian_greedy(::Type{Gtype}, ::Type{T}, a::T, b::T, Lt::Int, Ginit::AbstractVector{Gtype}, apply_op, nb_terms::Int;
                                        maxiter::Int = 1000, verbose::Bool=false, fullverbose::Bool=false) where{Gtype<:AbstractWavePacket, T<:AbstractFloat}
    
    verbose = verbose || fullverbose
    psize = param_size(Gtype)

    n0 = length(Ginit)
    G0_ = zeros(eltype(Ginit), n0 + nb_terms)
    @views G0_[1 : n0] = Ginit

    Tf = typeof(apply_op(a, zero(Gtype)))
    Gf_ = zeros(Tf, nb_terms, Lt)
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

    for iter=1:nb_terms
        verbose && println("Computing term $iter...")
        G[iter, :], _ = schrodinger_best_gaussian(Gtype, T, a, b, Lt, G0_[1 : n0 + iter - 1], apply_op,
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
        F[iter] = @views schrodinger_gaussian_residual_linear_part(Gtype, a, b, Lt, Ginit, apply_op, Gf, Gg, X[:, iter])

        # Λ[1:iter] = T(-0.5) .* (GramMatrix[1:iter, 1:iter] \ conj.(F[1:iter]))
        Λ[1:iter] = ones(iter)
        res = real(dot(Λ[1:iter], GramMatrix[1:iter, 1:iter], Λ[1:iter]) - 2 * dot(F[1:iter], Λ[1:iter]) + C0)
        verbose && println("Residual = $res")
        res_list[iter] = res
        # println("Λ = ", Λ[1:iter])

        # display(GramMatrix)

        #Fills the right member
        for j=1:iter
            for k=1:Lt
                t = a + (k-1)/(Lt-1) * (b - a)
                Gf_[j, k] = -Λ[j] * apply_op(t, G[j, k])
                Gg_[j, k] = -Λ[j] * G[j, k]
            end
            G0_[n0 + j] = -Λ[j] * G[j, 1]
        end

        # res0 = gaussian_approx_residual_constant_part(G0_)
        # println("Residual 0 = $res0")

        verbose && println()
    end

    for iter=1:nb_terms
        for k=1:Lt
            G[iter, k] = G[iter, k]
        end
    end

    return G, res_list
end