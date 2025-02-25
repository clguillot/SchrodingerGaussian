using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings

import LinearAlgebra.BLAS

include("apply_op.jl")

# include("reference_solution.jl")

function test_schrodinger_multidim_greedy(a::T, b::T, Lt, nb_terms::Int, newton_nb_iter::Int, ::Type{T}, plot_resut) where{T<:AbstractFloat}

    D = 2
    Gtype = GaussianWavePacket{D, Complex{T}, Complex{T}, T, T}
    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        λ0 = complex(1.0)
        z0 = SVector{D}(fill(complex(1.0), D))
        q0 = SVector{D}([1.0 ; fill(0.0, D-1)])
        p0 = SVector{D}([-1.0 ; fill(0.0, D-1)])
        G0 = [GaussianWavePacket(λ0, z0, q0, p0)]

        # Gv = Gaussian1D(1.0, 1.0, 0.0)
        # v(x) = Gv(x)

        G_list, res_list = schrodinger_gaussian_greedy(Gtype, T, a, b, Lt, G0, apply_op, nb_terms; maxiter=newton_nb_iter, verbose=true, fullverbose=false)

        # M = 20.0
        # Lx = 4096
        # hx = 2*M / (Lx+1)
        # U = schrodinger_sine(a, b, Lt, G0, v, M, Lx)

        # if plot_resut
        #     x_list = T.(-10:0.02:10)
        #     t_list = zeros(Lt)
        #     norm_list = zeros(Lt)
        #     g = @gif for k in 1:Lt
        #         t = a + (k-1) * (b-a)/(Lt-1)
        #         G = zeros(Gtype, nb_terms)
        #         for j=1:nb_terms
        #             G[j] = inv_fourier(unitary_product(2*t, fourier(G_list[j, k])))
        #         end
        #         t_list[k] = t
        #         norm_list[k] = norm_L2(G)
        #         fgx = G.(x_list)
        #         fx = abs2.(fgx)
        #         # fx_re = real.(fgx)
        #         # fx_im = imag.(fgx)
        #         fx_v = v.(x_list)

        #         f_ref = zeros(length(x_list))
        #         for j in eachindex(x_list)
        #             x = x_list[j]
        #             μ = complex(0.0)
        #             for p in 1:Lx
        #                 μ += U[p, k] * sin(π * p * (x + M) / (2*M))
        #             end
        #             f_ref[j] = abs2(μ)
        #         end
        #         plot(x_list, [fx, fx_v, f_ref], legend=:none, ylims=(-0.4, 2.0))
        #     end fps=30 every (max(round(Int, Lt / (30 * (b-a))), 1))
        #     display(g)

        #     println("res_list = ", res_list)

        #     p = plot()
        #     plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        #     plot!(p, t_list, fill(norm_L2(G0[1]), Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
        #     display(p)
        #     # savefig("norm.pdf")

        #     # p = plot()
        #     # for k in [Lt]
        #     #     t = a + (k-1) * (b-a)/(Lt-1)
        #     #     G = zeros(GT, nb_terms)
        #     #     for j=1:nb_terms
        #     #         G[j] = inv_fourier(unitary_product(2*t, fourier(G_list[j, k])))
        #     #     end
        #     #     fgx = G.(x_list)
        #     #     fx = abs2.(fgx)
        #     #     plot!(p, x_list, fx; label=LaTeXString("\$|ψ($(round(Int, t)), x)|^2\$"), ylims=(-0.4, 2.0))
        #     # end
        #     # plot!(p, x_list, v.(x_list); label=LaTeXString("\$v(x)\$"))
        #     # display(p)
        #     # savefig("state.pdf")
        # end
    
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end

    return nothing
end