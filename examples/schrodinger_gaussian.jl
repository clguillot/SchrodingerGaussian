using SchrodingerGaussian
using HermiteWavePackets
using Printf

import LinearAlgebra.BLAS

function test_schrodinger_gaussian(a::T, b::T, Lt, newton_nb_iter::Int, ::Type{T}, plot_resut) where{T<:AbstractFloat}

    GT = GaussianWavePacket1D{Complex{T}, Complex{T}, T, T}
    blas_nb_threads = BLAS.get_num_threads()

    try
        BLAS.set_num_threads(1)

        G0 = [GaussianWavePacket1D(complex(1.0), complex(1.0), 2.0, -1.0)]

        Gv = GaussianWavePacket1D(2.0, 1.0, 0.0, 0.0)
        function apply_op(t, Gop, G)
            G_ = Gop * inv_fourier(unitary_product(2*t, fourier(G)))
            return inv_fourier(unitary_product(-2*t, fourier(G_)))
        end
        
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

        G_list, val = schrodinger_best_gaussian(a, b, Lt, G0, apply_op, Gv, Gf, Gg, sqrt(eps(T)); maxiter=newton_nb_iter, verbose=false)
        println("Residual = $val")

        # if plot_resut
        #     x_list = T.(-12:0.05:12)
        #     q_list = zeros(length(G_list))
        #     p_list = zeros(length(G_list))
        #     norm_list = zeros(length(G_list))
        #     for k in eachindex(G_list)
        #         t = a + (k-1) * (b-a)/(Lt-1)
        #         Geit = Gaussian{T}(1.0, 2im*t, 0.0, 0.0)
        #         G_real = inv_fourier(Geit * fourier(G_list[k]))
        #         q_list[k] = G_real.q
        #         p_list[k] = G_real.p
        #         norm_list[k] = sqrt(dot_L2(G_real, G_real))
        #         fx = [abs2(G_real(x)) for x in x_list]
        #         fx_re = [real(G_real(x)) for x in x_list]
        #         fx_im = [imag(G_real(x)) for x in x_list]
        #         fx_v = [real(Gv(x)) for x in x_list]
        #         if true || k==Lt
        #             display(plot(x_list, [fx, fx_re, fx_im, fx_v], legend=:none, ylims=(-1.2, 1.2)))
        #         end
        #     end

        #     display(plot(q_list; label="Position"))
        #     display(plot(p_list; label="Momentum"))
        #     display(plot(norm_list; label="L2 Norm"))
        # end

        println("Test application :")
        display(apply_op(1.0, Gv, G0[1]))
        
        Gend = G_list[1]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)

        Gend = G_list[end]
        @printf("(%.12f%+.12fi)exp(-(%.12f%+.12fi)/2(x%+.12f)^2%+.12fxi)\n", real(Gend.λ), imag(Gend.λ), real(Gend.z), imag(Gend.z), -Gend.q, Gend.p)
    
    finally
        BLAS.set_num_threads(blas_nb_threads)
    end
end