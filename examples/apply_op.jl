using HermiteWavePackets

function apply_op(t::T, G::Gtype) where{T<:Real, Gtype<:AbstractWavePacket1D}
    G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    Gv = GaussianWavePacket1D(1.0, 1.0, 0.0, 0.0)
    G2 = Gv * G1
    # P = SVector(zero(T), zero(T), -one(T), zero(T), one(T))
    # G2 = polynomial_product(zero(T), P, HermiteWavePacket1D(G1))
    return inv_fourier(unitary_product(-2*t, fourier(G2)))
end