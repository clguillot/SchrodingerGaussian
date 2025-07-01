

function schrodinger_gaussian_polynomial_greedy(::Type{N}, a::T, b::T, Lt::Int,
                Ginit::AbstractWavePacket{D}, apply_op, nb_terms::Int;
                maxiter::Int = 1000, verbose::Bool=false, fullverbose::Bool=false) where{D, N, T<:AbstractFloat}
    
    verbose = verbose || fullverbose
    Gtype = GaussianWavePacket{D, Complex{T}, Complex{T}, T, T}
    L = prod(N.parameters)
    Htype = HermiteWavePacket{D, N, Complex{T}, Complex{T}, T, T, L}
    idN = SMatrix{L, L}(diagm(ones(SVector{L, T})))
    h = (b-a)/(Lt-1)

    G0_ = zeros(Htype, nb_terms)

    Gf_ = fill(apply_op(a, zero(Htype)), nb_terms, Lt)
    Gf = zeros(Gtype, 0, Lt)

    Gg_ = zeros(Htype, nb_terms, Lt)
    Gg = zeros(Gtype, 0, Lt)

    H = zeros(Htype, nb_terms, Lt)

    abs_tol = sqrt(eps(T)) #T(1e-4)

    A = BlockBandedMatrix(Diagonal(zeros(Complex{T}, Lt * L)),
            fill(L, Lt), fill(L, Lt), (1,1))
    X = zeros(Complex{T}, L, Lt)
    X_flat = reshape(X, L*Lt)
    Y = zeros(Complex{T}, L, Lt)
    Y_flat = reshape(Y, L*Lt)
    cfg = SchBestGaussianCFG(Gtype, T, Lt)

    res = norm2_L2(Ginit)
    res_list = zeros(T, nb_terms)

    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        for iter=1:nb_terms
            verbose && println("Computing term $iter...")
            G, _ = @views schrodinger_best_gaussian(Gtype, T, a, b, Lt, Ginit + WavePacketSum(G0_[1 : iter-1]), apply_op,
                    Gf_[1:iter-1, :], Gg_[1:iter-1, :], abs_tol, cfg;
                    maxiter=maxiter, verbose=fullverbose)
            
            # Fills the Gram matrix A
            fill!(A, zero(T))
            fill!(Y, zero(T))
            @threads for k in 1:Lt
                tk = a + (k-1)*h
                Gk = G[k]
                for i in 1:L
                    Hi = HermiteWavePacket(SArray{N}(idN[:, i]...), Gk.z, Gk.q, Gk.p)
                    HHi = apply_op(tk, Hi)

                    # Right member
                    Y[i, k] = zero(eltype(Y))
                    for l in max(1,k-1):min(Lt,k+1)
                        Y[i, k] += (b - a) * schrodinger_gaussian_cross_residual(h, Lt, k, l, Hi, WavePacketSum(@view Gg_[1:iter-1, l]), HHi, WavePacketSum(@view Gf_[1:iter-1, l]))
                    end
                    if k == 1
                        Y[i, k] += dot_L2(Hi, Ginit + WavePacketSum(@view G0_[1:iter-1]))
                    end

                    # Mass matrix
                    for l in k:min(Lt,k+1)
                        tl = a + (l-1)*h
                        Gl = G[l]
                        for j in 1:L
                            Hj = HermiteWavePacket(SArray{N}(idN[:, j]...), Gl.z, Gl.q, Gl.p)
                            HHj = apply_op(tl, Hj)

                            μ = (b - a) * schrodinger_gaussian_cross_residual(h, Lt, k, l, Hi, Hj, HHi, HHj)
                            if k == l && i == j
                                @views A[Block(k, l)][i, j] = real(μ)
                            else
                                @views A[Block(k, l)][i, j] = μ
                                @views A[Block(l, k)][j, i] = conj(μ)
                            end
                        end
                    end
                end
                if k == 1
                    @views A[Block(k, k)] .+= idN
                end
            end

            block_tridiagonal_cholesky_solver_static!(X_flat, A, Y_flat, Val(L))
            # X_flat .= A \ Y_flat
            res += real(dot(X_flat, A, X_flat) - 2 * dot(X_flat, Y_flat))
            res_list[iter] = res
            verbose && println("Residual = $res")

            # #Fills the right member
            for k=1:Lt
                Gk = G[k]
                g = HermiteWavePacket(SArray{N}(reshape((@view X[:, k]), N.parameters...)), Gk.z, Gk.q, Gk.p)
                H[iter, k] = g
                t = a + (k-1)*h
                Gf_[iter, k] = - apply_op(t, g)
                Gg_[iter, k] = - g
                if k == 1
                    G0_[iter] = - g
                end
            end

            verbose && println()
        end
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end

    return H, res_list
end

#=

=#
function schrodinger_gaussian_greedy_polynomial_timestep(::Type{N}, a::T, b::T, Lt::Int, nb_timesteps::Int,
                Ginit::AbstractWavePacket{D}, apply_op, nb_greedy_terms::Int;
                progressbar::Bool=false, maxiter::Int = 1000, verbose::Bool=false, fullverbose::Bool=false) where{T<:AbstractFloat,D,N}

    Gtype = GaussianWavePacket{D,Complex{T},Complex{T},T,T}
    L = prod(N.parameters)
    Htype = HermiteWavePacket{D, N, Complex{T}, Complex{T}, T, T, L}

    res = zero(T)
    G = zeros(Htype, nb_greedy_terms, Lt)
    lt = fld(Lt, nb_timesteps)
    h = (b-a) / (Lt-1)
    for p in (progressbar ? ProgressBar(1:nb_timesteps) : 1:nb_timesteps)
        k1 = (p-1)*lt + 1
        k2 = (p == nb_timesteps) ? Lt : p*lt + 1
        a_ = a + (k1-1)*h
        b_ = a + (k2-1)*h
        lt_ = k2 - k1 + 1
        G0_ = (p == 1) ? Ginit : WavePacketSum(@view G[:, k1])
        G_block, res_list = schrodinger_gaussian_polynomial_greedy(N, a_, b_, lt_, G0_, apply_op, nb_greedy_terms; maxiter, verbose, fullverbose)
        @views G[:, k1:k2] .= G_block
        res += sqrt(res_list[end])
    end

    return G, res
end