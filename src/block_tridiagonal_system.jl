
#=
    Solves the linear system
        AX = Y
    where
        A =
        D₁   B₁ᵀ
        B₁   D₂   B₂ᵀ
          ⋱    ⋱   ⋱
            Bₙ₋₂  Dₙ₋₁ Bₙ₋₁ᵀ
                  Bₙ₋₁ Dₙ
    using a block Cholesky factorization, assuming that all blocks in A are of size block_size
    
    - X and Y should not reference the same data

    (see https://www.intel.com/content/www/us/en/docs/onemkl/cookbook/2021-4/factor-block-tridiag-symm-pos-def-matrices.html)
=#
struct BlockCholeskyStaticConfig{T<:Real}
    #Matrix buffers
    Id::Diagonal{T, Vector{T}}
    V::Matrix{T}
    W::Matrix{T}

    #Vector buffers
    x::Vector{T}

    #Block buffers
    U::Vector{UpperTriangular{T, Matrix{T}}} #Diagonal blocks
    B_up::Vector{Matrix{T}}  #Upper diagonal blocks
end
function BlockCholeskyStaticConfig(Y::Vector{T}, ::Val{block_size}) where{T<:Real, block_size}
    Id = Diagonal(ones(T, block_size))
    V = zeros(T, block_size, block_size)
    W = zeros(T, block_size, block_size)

    x = zeros(T, block_size)

    nb_blocks = length(Y) ÷ block_size
    U = [UpperTriangular(zeros(T, block_size, block_size)) for _=1:nb_blocks]
    B_up = [zeros(T, block_size, block_size) for _=1:nb_blocks-1]

    return BlockCholeskyStaticConfig(Id, V, W, x, U, B_up)
end
function block_tridiagonal_cholesky_solver_static!(X::Vector{T}, A::BlockBandedMatrix{T}, Y::Vector{T}, ::Val{block_size}, cfg::BlockCholeskyStaticConfig=BlockCholeskyStaticConfig(Y, Val(block_size))) where{T<:Real, block_size}
    (Lx, Ly) = size(A)
    nb_blocks = Lx ÷ block_size
    if Lx <= 0
        throw(DimensionMismatch("Cannot solve an empty linear system"))
    end
    if Lx != Ly
        throw(DimensionMismatch("Cannot solve non square linear system"))
    end
    if Lx != length(X) || Ly != length(Y)
        throw(DimensionMismatch("Linear system has incompatible dimensions"))
    end
    if blockbandwidths(A) != (1, 1)
        throw(DimensionMismatch("The matrix is not block tridiagonal"))
    end
    if Lx % block_size != 0
        throw(DimensionMismatch("The dimension of the system must be a multiple of block_size"))
    end

    @views A1 = SMatrix{block_size, block_size}(A[Block(1, 1)])
    C1 = cholesky(A1).U
    ldiv!(cfg.U[1], C1, cfg.Id)
    for k=1:nb_blocks-1
        mul!(cfg.B_up[k], cfg.U[k]', (@view A[Block(k, k+1)]))
        mul!(cfg.W, cfg.B_up[k]', cfg.B_up[k])
        cfg.W .= (@view A[Block(k+1, k+1)]) .- cfg.W
        Ck = cholesky(SMatrix{block_size, block_size}(cfg.W)).U
        ldiv!(cfg.U[k+1], Ck, cfg.Id)
    end

    Yb = reshape(Y, block_size, nb_blocks)
    Xb = reshape(X, block_size, nb_blocks)

    #Forward sweep (solves lower triangular system)
    mul!((@view Xb[:, 1]), cfg.U[1]', (@view Yb[:, 1]))
    for k=1:nb_blocks-1
        mul!(cfg.x, cfg.B_up[k]', (@view Xb[:, k]))
        cfg.x .= (@view Yb[:, k+1]) .- cfg.x
        mul!((@view Xb[:, k+1]), cfg.U[k+1]', cfg.x)
    end

    #Backward sweep (solves upper triangular system)
    mul!(cfg.x, cfg.U[end], (@view Xb[:, end]))
    copy!((@view Xb[:, end]), cfg.x)
    for k=nb_blocks:-1:2
        mul!(cfg.x, cfg.B_up[k-1], (@view Xb[:, k]))
        cfg.x .= (@view Xb[:, k-1]) .- cfg.x
        mul!((@view Xb[:, k-1]), cfg.U[k-1], cfg.x)
    end
    
    return X
end

# #=
#     Solves the linear system
#         AX = Y
#     where
#         A = 
#         D₁   C₁
#         B₁   D₂   C₂
#           ⋱    ⋱   ⋱
#             Bₙ₋₂  Dₙ₋₁ Cₙ₋₁
#                   Bₙ₋₁ Dₙ
#     using a block LU factorization

#     - X and Y should not reference the same data

#     (see https://www.sciencedirect.com/science/article/pii/S0377042706001269#bib22)
# =#
# function block_tridiagonal_lu_solver!(X::Vector{T}, A::BlockBandedMatrix{T}, Y::Vector{T}) where{T<:Real}
#     (Lx, Ly) = size(A)
#     if Lx <= 0
#         throw(DimensionMismatch("Cannot solve an empty linear system"))
#     end
#     if Lx != Ly
#         throw(DimensionMismatch("Cannot solve non square linear system"))
#     end
#     if Lx != length(X) || Ly != length(Y)
#         throw(DimensionMismatch("Linear system with incompatible dimensions"))
#     end
#     if blockbandwidths(A) != (1, 1)
#         throw(DimensionMismatch("The matrix is not block tridiagonal"))
#     end

#     lay_A = A.block_sizes
#     nb_blocks = length(lay_A.u)
    
#     #=
#         A = (Σ + L)Σ⁻¹(Σ + U)
#           = (1 + LΣ⁻¹)(Σ + U)
#           = L_(Σ + U)   where L_ = (1 + LΣ⁻¹)
#     =#
#     Σ = Vector{Matrix{T}}(undef, nb_blocks)
#     Σ_inv = Vector{Matrix{T}}(undef, nb_blocks)
#     L_ = Vector{Matrix{T}}(undef, nb_blocks-1)
#     U = Vector{Matrix{T}}(undef, nb_blocks-1)
#     block_start = zeros(Int, nb_blocks + 1) #Tells where the k-th block starts

#     block_start[1] = 1
#     Σ[1] = A[Block(1, 1)]
#     Σ_inv[1] = Σ[1]^(-1)
#     @views for k=1:nb_blocks-1
#         block_start[k+1] = block_start[k] + size(Σ[k], 1)
#         Σ[k+1] = A[Block(k+1, k+1)] - A[Block(k+1, k)] * Σ_inv[k] * A[Block(k, k+1)]
#         Σ_inv[k+1] = Σ[k+1]^(-1)
#         L_[k] = A[Block(k+1, k)] * Σ_inv[k]
#         U[k] = A[Block(k, k+1)]
#     end
#     if nb_blocks > 1
#         block_start[end] = block_start[end-1] + size(Σ[end], 1)
#     end

#     #Forward sweep (solves lower triangular system)
#     @views X[1 : block_start[2] - 1] .= Y[1 : block_start[2] - 1]
#     @views for k=1:nb_blocks-1
#         bs = block_start[k+1]
#         be = block_start[k+2] - 1
#         mul!(X[bs : be], L_[k], X[block_start[k] : block_start[k+1] - 1])
#         @. X[bs : be] = Y[bs : be] - X[bs : be]
#     end

#     #Backward sweep (solves upper triangular system)
#     @views X[block_start[end-1] : block_start[end] - 1] .=
#                 Σ_inv[end] * X[block_start[end-1] : block_start[end] - 1]
#     @views for k=nb_blocks:-1:2
#         bs = block_start[k-1]
#         be = block_start[k] - 1
#         V = X[bs : be] .- U[k-1] * X[block_start[k] : block_start[k+1] - 1]
#         mul!(X[bs : be], Σ_inv[k-1], V)
#     end
    
#     return X
# end

# function block_tridiagonal_lu_solver(A::BlockBandedMatrix{T}, Y::Vector{T}) where{T<:Real}
#     return block_tridiagonal_lu_solver!(similar(Y), A, Y)
# end

# #=
#     Solves the linear system
#         AX = Y
#     where
#         A =
#         D₁   B₁ᵀ
#         B₁   D₂   B₂ᵀ
#           ⋱    ⋱   ⋱
#             Bₙ₋₂  Dₙ₋₁ Bₙ₋₁ᵀ
#                   Bₙ₋₁ Dₙ
#     using a block Cholesky factorization

#     - X and Y should not reference the same data

#     (see https://www.intel.com/content/www/us/en/docs/onemkl/cookbook/2021-4/factor-block-tridiag-symm-pos-def-matrices.html)
# =#
# function block_tridiagonal_cholesky_solver!(X::Vector{T}, A::BlockBandedMatrix{T}, Y::Vector{T}) where{T<:Real}
#     (Lx, Ly) = size(A)
#     if Lx <= 0
#         throw(DimensionMismatch("Cannot solve an empty linear system"))
#     end
#     if Lx != Ly
#         throw(DimensionMismatch("Cannot solve non square linear system"))
#     end
#     if Lx != length(X) || Ly != length(Y)
#         throw(DimensionMismatch("Linear system with incompatible dimensions"))
#     end
#     if blockbandwidths(A) != (1, 1)
#         throw(DimensionMismatch("The matrix is not block tridiagonal"))
#     end

#     lay_A = A.block_sizes
#     nb_blocks = length(lay_A.u)
#     U = Vector{UpperTriangular{T, Matrix{T}}}(undef, nb_blocks)
#     B_up = Vector{Matrix{T}}(undef, nb_blocks - 1)
#     block_start = zeros(Int, nb_blocks + 1) #Tells where the k-th block starts

#     block_start[1] = 1
#     @views U[1] = cholesky(A[Block(1, 1)]).U^(-1)
#     @views for k=1:nb_blocks-1
#         block_start[k+1] = block_start[k] + size(U[k], 1)
#         B_up[k] = U[k]' * A[Block(k, k+1)]
#         D = A[Block(k+1, k+1)] .- B_up[k]' * B_up[k]
#         U[k+1] = cholesky(D).U^(-1)
#     end
#     if nb_blocks > 1
#         block_start[end] = block_start[end-1] + size(U[end], 1)
#     end

#     #Forward sweep (solves lower triangular system)
#     @views X[1 : block_start[2] - 1] .= U[1]' * Y[1 : block_start[2] - 1]
#     @views for k=1:nb_blocks-1
#         bs = block_start[k+1]
#         be = block_start[k+2] - 1
#         mul!(X[bs : be], B_up[k]', X[block_start[k] : block_start[k+1] - 1])
#         @. X[bs : be] = Y[bs : be] - X[bs : be]
#         X[bs : be] .= U[k+1]' * X[bs : be]
#     end

#     #Backward sweep (solves upper triangular system)
#     @views X[block_start[end-1] : block_start[end] - 1] .=
#                 U[end] * X[block_start[end-1] : block_start[end] - 1]
#     @views for k=nb_blocks:-1:2
#         bs = block_start[k-1]
#         be = block_start[k] - 1
#         V = X[bs : be] .- B_up[k-1] * X[block_start[k] : block_start[k+1] - 1]
#         mul!(X[bs : be], U[k-1], V)
#     end
    
#     return X
# end

# function block_tridiagonal_cholesky_solver(A::BlockBandedMatrix{T}, Y::Vector{T}) where{T<:Real}
#     return block_tridiagonal_cholesky_solver!(similar(Y), A, Y)
# end