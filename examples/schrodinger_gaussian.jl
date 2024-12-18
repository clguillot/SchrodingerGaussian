using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots

import LinearAlgebra.BLAS

function apply_op(t::T, G::Gtype) where{T<:Real, Gtype<:AbstractWavePacket1D}
    G1 = inv_fourier(unitary_product(2*t, fourier(G)))
    # Gv = GaussianWavePacket1D(2.0, 1.0, 0.0, 0.0)
    # G2 = Gv * G1
    P = SVector(zero(T), zero(T), -one(T), zero(T), one(T))
    G2 = polynomial_product(zero(T), P, HermiteWavePacket1D(G1))
    return inv_fourier(unitary_product(-2*t, fourier(G2)))
end

function test_schrodinger_gaussian(a::T, b::T, Lt, newton_nb_iter::Int, ::Type{T}, plot_resut) where{T<:AbstractFloat}

    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        G0 = [GaussianWavePacket1D(complex(0.5), complex(8.0), 1/sqrt(2.0), 0.0)]
        
        Gf = zeros(GT, 0, Lt)
        # for k in eachindex(Gf)
        #     t = a + (k-1)/(Lt-1)*(b-a)
        #     Gf[1, k] = Gaussian{T}(0.5*exp(t), 1.0, 5.0, -1.0)
        # end
        Gg = zeros(GT, 0, Lt)
        # for k in eachindex(Gg)
        #     t = a + (k-1)/(Lt-1)*(b-a)
        #     Gg[1, k] = Gaussian{T}(0.5*exp(t), 1.0, 5.0, -1.0)
        # end

        v(x) = x^4 - x^2

        G_list, val = schrodinger_best_gaussian(a, b, Lt, G0, apply_op, Gf, Gg, sqrt(eps(T)); maxiter=newton_nb_iter, verbose=false)
        println("Residual = $val")

        if plot_resut
            x_list = T.(-3:0.02:3)
            t_list = zeros(Lt)
            q_list = zeros(Lt)
            p_list = zeros(Lt)
            norm_list = zeros(Lt)
            g = @gif for k in 1:Lt
                t = a + (k-1) * (b-a)/(Lt-1)
                G = inv_fourier(unitary_product(2*t, fourier(G_list[k])))
                q_list[k] = G.q
                p_list[k] = G.p
                t_list[k] = t
                norm_list[k] = norm_L2(G)
                fgx = G.(x_list)
                fx = abs2.(fgx)
                # fx_re = real.(fgx)
                # fx_im = imag.(fgx)
                fx_v = v.(x_list)
                plot(x_list, [fx, fx_v], legend=:none, ylims=(-0.4, 1.2))
            end fps=60 every (cld(Lt, 60))

            display(g)
            display(plot(q_list; label="Position"))
            display(plot(p_list; label="Momentum"))
            display(plot(norm_list; label="L2 Norm"))
        end

        println("Test application :")
        display(apply_op(1.0, G0[1]))
        
        Gend = G_list[1]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)

        Gend = G_list[end]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)
    
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end
end