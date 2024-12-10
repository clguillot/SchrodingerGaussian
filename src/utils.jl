#=
    Finite element factors
=#

#=
    Returns
        ∫dt ζₖ(t)ζₗ(t)
    where the (ζₖ)ₖ are the P1 finite elements such that ζₖ(lh)=δₖₗ
=#
@inline function fe_m_factor(h::Real, k::Int, l::Int)
    if k == l
        return 2*h/3
    elseif abs(k-l) == 1
        return h/6
    else
        return zero(h)
    end
end

#=
    Returns
        ∫dt ζₖ'(t)ζₗ'(t)
    where the (ζₖ)ₖ are the P1 finite elements such that ζₖ(lh)=δₖₗ
=#
@inline function fe_k_factor(h::Real, k::Int, l::Int)
    if k == l
        return 2/h
    elseif abs(k-l) == 1
        return -1/h
    else
        return zero(1/h)
    end
end

#=
    Returns
        ∫dt ζₖ'(t)ζₗ(t)
    where the (ζₖ)ₖ are the P1 finite elements such that ζₖ(lh)=δₖₗ
=#
@inline function fe_l_factor(h::T, k::Int, l::Int) where{T<:Real}
    if l == k+1
        return T(-1/2)
    elseif l == k-1
        return T(1/2)
    else
        return T(0)
    end
end

#=
    Returns
        ∫₍₀,ₕ₎ dt ζ₀'(t)ζ₀(t)
    where the (ζₖ)ₖ are the P1 finite elements such that ζₖ(lh)=δₖₗ
=#
@inline function fe_l_half_factor(h::T) where{T<:Real}
    return T(-1/2)
end

#=
    Gaussian packing and unpacking
=#
# Returns the size of a vector needed to pack and unpack GaussianWavePacket1D
const gaussian_param_size::Int = 6
param_size(::GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}) where T = 6
param_size(::Type{GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}}) where T = 6

#=
    Packing a GaussianWavePacket1D into an AbstractVector
    The values are packed from index idx to idx + param_size(G)
=#
@inline function pack_gaussian_parameters!(X::AbstractVector{T}, G::GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}, idx::Int=1) where T
    X[idx] = real(G.λ)
    X[idx + 1] = imag(G.λ)
    X[idx + 2] = real(G.z)
    X[idx + 3] = imag(G.z)
    X[idx + 4] = G.q
    X[idx + 5] = G.p
    return X
end

@inline function pack_gaussian_parameters(G::GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}) where T
    X = zeros(T, param_size(G))
    return pack_gaussian_parameters!(X, G)
end

#=
    Unpacking a GaussianWavePacket1D from an AbstractVector
    The values are extracted from index idx to idx + param_size(G)
=#
@inline function unpack_gaussian_parameters(X::AbstractVector{T}, idx::Int=1) where{T<:Real}
    return GaussianWavePacket1D(complex(X[idx], X[idx+1]), complex(X[idx+2], X[idx+3]), X[idx+4], X[idx+5])
end

#=
    Unpacking the parameter z from an AbstractVector
=#
@inline function unpack_gaussian_parameter_z(X::AbstractVector{T}, idx::Int=1) where{T<:Real}
    return complex(X[idx+2], X[idx+3])
end