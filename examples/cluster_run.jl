using SchrodingerGaussian
using HermiteWavePackets
using StaticArrays
using JLD2

function apply_op(t, G::AbstractWavePacket{D}) where D
    G1 = inv_fourier(unitary_product(fourier(G), (@SVector fill(2*t, D))))
    Gv = Gaussian(1.0, (@SVector fill(1.0, D)))
    G2 = Gv * G1
    return inv_fourier(unitary_product(fourier(G2), (@SVector fill(2*t, D))))
end

function test_cluster(a::T, b::T, Lt, nb_terms::Int, newton_nb_iter::Int) where{T<:AbstractFloat}

    Gtype = GaussianWavePacket{3, Complex{T}, Complex{T}, T, T}

    G0 = GaussianWavePacket(complex(1.0), complex.(SVector(1.0, 1.0, 1.0)), SVector(6.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0))
    G0 = G0 / norm_L2(G0)

    time_elapsed = @elapsed G_list, res_list = schrodinger_gaussian_greedy(Gtype, T, a, b, Lt, G0, apply_op, nb_terms;greedy_orthogonal=false, maxiter=newton_nb_iter, verbose=false, fullverbose=false)

    return G0, G_list, res_list, time_elapsed
end

G0, G_list, res_list, time_elapsed = test_cluster(0.0, 10.0, 10, 1, 1)
G0, G_list, res_list, time_elapsed = test_cluster(0.0, 10.0, 1000, 50, 1000)
@save "data.jld2" G0 G_list res_list time_elapsed