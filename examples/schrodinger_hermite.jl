using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings



include("reference_solution.jl")

function plot_schrodinger_hermite(pb::GreedyDiscretization{T}, ::Type{N}, G0::AbstractWavePacket{D}, V::AbstractWavePacket{D}; plot_result=true) where{T, D, N}

    function propag(t, G::AbstractWavePacket{D}) where D
        return inv_fourier(unitary_product(fourier(G), SVector{D}(ntuple(_ -> 2*t, D)...)))
    end
    function apply_op(t, G)
        return propag(-t, V * propag(t, G))
    end
    G_list, res_list = schrodinger_greedy_hermite(pb, N, G0, apply_op)

    M = 30.0
    Lx = 4096
    U = schrodinger_sine(pb.t0, pb.tf, pb.Lt, G0, V, M, Lx)

    if plot_result
        x_list = T.(-10:0.02:10)
        t_list = zeros(pb.Lt)
        norm_list = zeros(pb.Lt)
        g = @gif for k in 1:pb.Lt
            t = pb.t0 + (k-1) * (pb.tf-pb.t0)/(pb.Lt-1)
            G = WavePacketSum{D}(propag.(t, G_list[:, k]))
            t_list[k] = t
            norm_list[k] = norm_L2(G)
            fgx = G.(x_list)
            fx = abs2.(fgx)
            fx_v = V.(x_list)

            f_ref = zeros(length(x_list))
            for j in eachindex(x_list)
                x = x_list[j]
                μ = complex(0.0)
                for p in 1:Lx
                    μ += U[p, k] * sin(π * p * (x + M) / (2*M))
                end
                f_ref[j] = abs2(μ)
            end
            plot(x_list, [fx, fx_v, f_ref], legend=:none, ylims=(-0.4, 2.0))
        end fps=30 every (max(round(Int, pb.Lt / (30 * (pb.tf-pb.t0))), 1))
        display(g)

        println("res_list = ", res_list)

        p = plot()
        plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(p, t_list, fill(norm_L2(G0), pb.Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
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

G0 = GaussianWavePacket(complex(1.0),complex(1.0),6.,-1.0)
G0 = G0 / norm_L2(G0)
pb = GreedyDiscretization(0., 5., 100, 10, 100, false)
V = Gaussian(1.5, 1.0, -2.0) + Gaussian(1.0, 1.0, 2.0)
plot_schrodinger_hermite(pb, Tuple{8}, G0, V)