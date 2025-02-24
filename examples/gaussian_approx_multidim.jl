using SchrodingerGaussian
using HermiteWavePackets
using StaticArrays

function test_gaussian_multidim_approx()

    λ1 = complex(1.0)
    z1 = SVector(complex(1.0), complex(0.5, 1.0), complex(2.0, -0.5))
    q1 = SVector(1.0, -2.3, 0.6)
    p1 = SVector(0.0, 0.2, 1.5)
    # Parameters for G2
    λ2 = complex(1.5)
    z2 = SVector(complex(1.2, 0.3), complex(0.8, 1.5), complex(1.5, -0.8))
    q2 = SVector(1.5, -1.8, 0.9)
    p2 = SVector(0.5, 0.3, 1.2)
    # Parameters for G3
    λ3 = complex(0.8)
    z3 = SVector(complex(0.9, 0.4), complex(0.7, 1.2), complex(1.8, -0.7))
    q3 = SVector(1.2, -2.0, 0.7)
    p3 = SVector(0.3, 0.4, 1.1)

    # Wave packets
    G1 = GaussianWavePacket(λ1, z1, q1, p1)
    G2 = GaussianWavePacket(λ2, z2, q2, p2)
    G3 = GaussianWavePacket(λ3, z3, q3, p3)
    
    # Perturbation
    δ = 1e-1
    λ_init = λ1 + δ * complex(rand(), rand())
    z_init = z1 + δ * ((@SVector rand(3)) + im * (@SVector rand(3)))
    q_init = q1 + δ * @SVector rand(3)
    p_init = p1 + δ * @SVector rand(3)
    G_init = GaussianWavePacket(λ_init, z_init, q_init, p_init)

    Gtype = GaussianWavePacket{3, ComplexF64, ComplexF64, Float64, Float64}

    G_approx = gaussian_approx(Gtype, Float64, [G1, G2, G3], G_init; rel_tol=1e-12, verbose=true)

    # display(G)
    display(G_approx)
end