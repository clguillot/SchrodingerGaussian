using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings

import LinearAlgebra.BLAS

include("reference_solution.jl")

function apply_op_polynomial(t, G::GaussianWavePacket)
    G1 = inv_fourier(unitary_product(fourier(G), SVector(2*t)))
    Gv = Gaussian(1.0, 1.0)
    G2 = Gv * G1
    return inv_fourier(unitary_product(fourier(G2), SVector(-2*t)))
end
function apply_op_polynomial(t, G::HermiteWavePacket)
    G1 = inv_fourier(unitary_product(fourier(G), SVector(2*t)))
    Gv = HermiteFct(Gaussian(1.0, 1.0))
    G2 = Gv * G1
    return inv_fourier(unitary_product(fourier(G2), SVector(-2*t)))
end

function test_schrodinger_greedy_polynomial(a::T, b::T, Lt, nb_terms::Int, newton_nb_iter::Int, ::Type{N}, ::Type{T}, plot_resut) where{T<:AbstractFloat, N}

    Gtype = GaussianWavePacket{1, Complex{T}, Complex{T}, T, T}

    # G0 = GaussianWavePacket1D(complex(1.0), complex(1.0), 6.0, -1.0)
    # G0 = inv(norm_L2(G0)) * G0
    G0 = GaussianWavePacket(complex(1.0), complex(1.0), 6.0, -1.0)
    G0 = G0 / norm_L2(G0)
    # G0 = [GaussianWavePacket1D(complex(0.5), complex(8.0), 1/sqrt(2.0), 0.0)]
    # G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 0.0, 0.0)]

    Gv = Gaussian(1.0, 1.0)
    v(x) = Gv(x)
    # v(x) = x^4 - x^2
    # Gv1 = Gaussian1D(1.0, 1.0, 2.0)
    # Gv2 = Gaussian1D(1.0, 1.0, -2.0)
    # v(x) = Gv1(x) + Gv2(x)

    G_list, res_list = schrodinger_gaussian1d_polynomial_greedy(Gtype, N, T, a, b, Lt, G0, apply_op_polynomial, nb_terms; maxiter=newton_nb_iter, verbose=true, fullverbose=false)

    # M = 30.0
    # Lx = 4096
    # U = schrodinger_sine(a, b, Lt, WavePacketArray(G0), v, M, Lx)

    if plot_resut
        x_list = T.(-10:0.02:10)
        t_list = zeros(Lt)
        norm_list = zeros(Lt)
        g = @gif for k in 1:Lt
            t = a + (k-1) * (b-a)/(Lt-1)
            G = WavePacketSum([inv_fourier(unitary_product(fourier(G_list[j, k]), SVector(2*t))) for j in 1:nb_terms])
            t_list[k] = t
            norm_list[k] = norm_L2(G)
            fgx = G.(x_list)
            fx = abs2.(fgx)
            fx_v = v.(x_list)

            # f_ref = zeros(length(x_list))
            # for j in eachindex(x_list)
            #     x = x_list[j]
            #     μ = complex(0.0)
            #     for p in 1:Lx
            #         μ += U[p, k] * sin(π * p * (x + M) / (2*M))
            #     end
            #     f_ref[j] = abs2(μ)
            # end
            plot(x_list, [fx, fx_v#=, f_ref=#], legend=:none, ylims=(-0.4, 2.0))
        end fps=30 every (max(round(Int, Lt / (30 * (b-a))), 1))
        display(g)

        println("res_list = ", res_list)

        p = plot()
        plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(p, t_list, fill(norm_L2(G0), Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
        display(p)
        # savefig("norm.pdf")

        # p = plot()
        # for k in [Lt]
        #     t = a + (k-1) * (b-a)/(Lt-1)
        #     G = zeros(GT, nb_terms)
        #     for j=1:nb_terms
        #         G[j] = inv_fourier(unitary_product(2*t, fourier(G_list[j, k])))
        #     end
        #     fgx = G.(x_list)
        #     fx = abs2.(fgx)
        #     plot!(p, x_list, fx; label=LaTeXString("\$|ψ($(round(Int, t)), x)|^2\$"), ylims=(-0.4, 2.0))
        # end
        # plot!(p, x_list, v.(x_list); label=LaTeXString("\$v(x)\$"))
        # display(p)
        # savefig("state.pdf")
    end
end