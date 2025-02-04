using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings

import LinearAlgebra.BLAS

include("apply_op.jl")

function test_schrodinger_greedy(a::T, b::T, Lt, nb_terms::Int, newton_nb_iter::Int, ::Type{T}, plot_resut) where{T<:AbstractFloat}

    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 6.0, -1.0)]
        # G0 = [GaussianWavePacket1D(complex(0.5), complex(8.0), 1/sqrt(2.0), 0.0)]
        # G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 0.0, 0.0)]
        
        Gf = zeros(GT, 0, Lt)
        # for k in eachindex(Gf)
        #     t = a + (k-1)/(Lt-1)*(b-a)
        #     Gf[1, k] = Gaussian{T}(0.5*exp(t), 1.0, 5.0, -1.0)
        # end

        Gv = Gaussian1D(1.0, 1.0, 0.0)
        v(x) = Gv(x)
        # v(x) = x^4 - x^2
        # Gv1 = Gaussian1D(1.0, 1.0, 2.0)
        # Gv2 = Gaussian1D(1.0, 1.0, -2.0)
        # v(x) = Gv1(x) + Gv2(x)

        G_list, res_list = schrodinger_gaussian_greedy(a, b, Lt, G0, apply_op, nb_terms; maxiter=newton_nb_iter, verbose=true, fullverbose=true)

        if plot_resut
            x_list = T.(-10:0.02:10)
            t_list = zeros(Lt)
            norm_list = zeros(Lt)
            g = @gif for k in 1:Lt
                t = a + (k-1) * (b-a)/(Lt-1)
                G = zeros(GT, nb_terms)
                for j=1:nb_terms
                    G[j] = inv_fourier(unitary_product(2*t, fourier(G_list[j, k])))
                end
                t_list[k] = t
                norm_list[k] = norm_L2(G)
                fgx = G.(x_list)
                fx = abs2.(fgx)
                # fx_re = real.(fgx)
                # fx_im = imag.(fgx)
                fx_v = v.(x_list)
                plot(x_list, [fx, fx_v], legend=:none, ylims=(-0.4, 2.0))
            end fps=30 every (max(round(Int, Lt / (30 * (b-a))), 1))
            display(g)

            println("res_list = ", res_list)

            p = plot()
            plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
            plot!(p, t_list, fill(norm_L2(G0[1]), Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
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

        println("Test application :")
        display(apply_op(1.0, G0[1]))
        
        Gend = G_list[1, 1]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)

        Gend = G_list[1, end]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)
    
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end
end