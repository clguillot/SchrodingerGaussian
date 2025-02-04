using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings
using ProgressBars

import LinearAlgebra.BLAS

include("apply_op.jl")

function test_schrodinger_greedy_timestep(a::T, b::T, Lt::Int, nb_steps::Int, nb_terms::Int, newton_nb_iter::Int, ::Type{T}, plot_resut) where{T<:AbstractFloat}

    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 6.0, -1.0)]
        # G0 = [GaussianWavePacket1D(complex(0.5), complex(8.0), 1/sqrt(2.0), 0.0)]
        # G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 0.0, 0.0)]

        Gv = Gaussian1D(1.0, 1.0, 0.0)
        v(x) = Gv(x)
        # v(x) = x^4 - x^2
        # Gv1 = Gaussian1D(1.0, 1.0, 2.0)
        # Gv2 = Gaussian1D(1.0, 1.0, -2.0)
        # v(x) = Gv1(x) + Gv2(x)

        G_list = zeros(GT, nb_terms, Lt)
        lt = fld(Lt, nb_steps)
        h = (b-a) / (Lt-1)
        for p in ProgressBar(1:nb_steps)
            k1 = (p-1)*lt + 1
            k2 = (p == nb_steps) ? Lt : p*lt + 1
            a_ = a + (k1-1)*h
            b_ = a + (k2-1)*h
            lt_ = k2 - k1 + 1
            G0_ = (p == 1) ? G0 : G_list[:, k1]
            G_list[:, k1:k2], _ = schrodinger_gaussian_greedy(a_, b_, lt_, G0_, apply_op, nb_terms; maxiter=newton_nb_iter, verbose=false)
        end

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

            p = plot()
            plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
            plot!(p, t_list, fill(norm_L2(G0[1]), Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
            display(p)
            # savefig("norm.pdf")
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