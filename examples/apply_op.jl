using HermiteWavePackets
using StaticArrays

function apply_op(t::T, G::Gtype) where{T<:Real, Gtype<:AbstractWavePacket1D}
    # G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    # Gv = GaussianWavePacket1D(1.0, 1.0, 0.0, 0.0)
    # G2 = Gv * G1
    # P = SVector(zero(T), zero(T), -one(T), zero(T), one(T))
    # G2 = polynomial_product(zero(T), P, HermiteWavePacket1D(G1))
    # return inv_fourier(unitary_product(-2*t, fourier(G2)))

    G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    Gv1 = GaussianWavePacket1D(1.0, 1.0, 2.0, 0.0)
    Gv2 = GaussianWavePacket1D(1.0, 1.0, -2.0, 0.0)
    H1 = inv_fourier(unitary_product(-2*t, fourier(Gv1 * G1)))
    H2 = inv_fourier(unitary_product(-2*t, fourier(Gv2 * G1)))
    return SVector(H1, H2)
end