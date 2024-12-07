
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