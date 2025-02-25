using HermiteWavePackets
using StaticArrays

# 1D operator
function apply_op(t::T, G::AbstractWavePacket1D) where{T<:Real}
    G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    Gv = GaussianWavePacket1D(1.0, 1.0, 0.0, 0.0)
    G2 = Gv * G1
    return inv_fourier(unitary_product(-2*t, fourier(G2)))
    # P = SVector(zero(T), zero(T), -one(T), zero(T), one(T))
    # G2 = polynomial_product(zero(T), P, HermiteWavePacket1D(G1))
    # return inv_fourier(unitary_product(-2*t, fourier(G2)))

    # G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    # Gv1 = GaussianWavePacket1D(1.0, 1.0, 2.0, 0.0)
    # Gv2 = GaussianWavePacket1D(1.0, 1.0, -2.0, 0.0)
    # H1 = inv_fourier(unitary_product(-2*t, fourier(Gv1 * G1)))
    # H2 = inv_fourier(unitary_product(-2*t, fourier(Gv2 * G1)))
    # return SVector(H1, H2)
end

# Generalized operator
function apply_op(t::T, G::GaussianWavePacket{D}) where{T<:Real, D}
    # Potential barrier
    Gv = GaussianWavePacket(1.0, (@SVector fill(1.0, D)), (@SVector zeros(D)), (@SVector zeros(D)))

    u = @SVector fill(2*t, D)
    G1 = inv_fourier(unitary_product(u, fourier(G)))
    G2 = Gv * G1
    return inv_fourier(unitary_product(-u, fourier(G2)))
end