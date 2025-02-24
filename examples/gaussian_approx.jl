using SchrodingerGaussian
using HermiteWavePackets

function test_gaussian_approx()

    # Real gaussian
    G1 = GaussianWavePacket1D(complex(1.0), complex(1.0), -1.0, 1.5)
    G2 = GaussianWavePacket1D(complex(1.0), complex(1.0), 0.0, 0.6)
    G3 = GaussianWavePacket1D(complex(1.0), complex(1.0), 1.0, -2.3)
    
    # Perturbation
    # λ_init = complex(rand(), rand())
    # z_init = complex(rand(), rand())
    # q_init = 2 * rand() - 1
    # p_init = 2 * rand() - 1
    # G_init = GaussianWavePacket1D(λ_init, z_init, q_init, p_init)
    G_init = -10.0 * G2

    Gtype = GaussianWavePacket1D{ComplexF64, ComplexF64, Float64, Float64}

    G_approx = gaussian_approx(Gtype, Float64, [G1, G2, G3], G_init; rel_tol=1e-12, verbose=true)

    # display(G)
    display(G_approx)
end