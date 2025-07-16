
#=
    Computes |G_X - Ginit[k]|^2 - |Ginit[k]|^2
=#
function gaussian_approx_residual(G::AbstractWavePacket, Ginit::AbstractWavePacket)
    return norm2_L2(G) - 2 * real(dot_L2(G, Ginit))
end

#=
    Returns |Ginit[k]|^2
=#
function gaussian_approx_residual_constant_part(Ginit::AbstractWavePacket)
    return norm2_L2(Ginit::AbstractWavePacket)
end

#Gradient
function gaussian_approx_gradient(::Type{Gtype}, Ginit::AbstractWavePacket{D},
                                    X::AbstractVector{T}) where{D, Gtype<:AbstractWavePacket{D}, T<:Real}
    psize = param_size(Gtype)
    X = SVector{psize}(X)

    function f(Y)
        G = unpack_gaussian_parameters(Gtype, Y)
        gaussian_approx_residual(G, Ginit)
    end
    
    return ForwardDiff.gradient(f, X)
end