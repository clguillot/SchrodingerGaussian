using SchrodingerGaussian
using HermiteWavePackets

function test_gaussian_approx()

    # Real gaussian
    λ = complex(1.0, 0.2)
    z = complex(1.23, -2.7)
    q = -2.3
    p = 5.6
    G = GaussianWavePacket1D(λ, z, q, p)
    
    # Perturbation
    δ = 0.01
    λ_init = λ + complex(rand() * δ, rand() * δ)
    z_init = z + complex(rand() * δ, rand() * δ)
    q_init = q + rand() * δ
    p_init = p + rand() * δ
    G_init = GaussianWavePacket1D(λ_init, z_init, q_init, p_init)

    G_approx = gaussian_approx([G], G_init; rel_tol=1e-12, verbose=true)

    display(G)
    display(G_approx)
end