using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings
using ProgressBars

import LinearAlgebra.BLAS

include("apply_op.jl")

function plot_greedy_timestep(dis::Discretization, pot::Potential, nb_timesteps; plot_result=true) 

    T = typeof(dis.t0)
    Gtype = typeof(dis.G0)

    G_list, res = schrodinger_gaussian_greedy_timestep(dis,pot,nb_timesteps)
    println("Residual = ", res)

    if plot_result
        x_list = T.(-10:0.02:10)
        t_list = zeros(dis.Nt)
        norm_list = zeros(dis.Nt)
        g = @gif for k in 1:dis.Nt
            t = dis.t0 + (k-1) * (dis.tf-dis.t0)/(dis.Nt-1)
            G = zeros(Gtype, dis.nb_g)
            for j=1:dis.nb_g
                G[j] = inv_fourier(unitary_product(fourier(G_list[j, k]), SVector(2*t)))
            end
            t_list[k] = t
            norm_list[k] = norm_L2(WavePacketSum(G))
            fgx = WavePacketSum(G).(x_list)
            fx = abs2.(fgx)
            fx_v = pot.(x_list)
            plot(x_list, [fx, fx_v], legend=:none, ylims=(-0.4, 2.0))
        end fps=30 every (max(round(Int, dis.Nt / (30 * (dis.tf-dis.t0))), 1))
        display(g)

        p = plot()
        plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(p, t_list, fill(norm_L2(G0), dis.Nt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
        display(p)
        # savefig("norm.pdf")
    end
end

G0 = GaussianWavePacket(complex(1.0),complex(1.0),6.,-1.0)
G0 = G0 / norm_L2(G0)
dis = Discretization(0.,10.,50,G0,10,100,false)
pot = Potential([1.5,1.0],[1.0,1.0],[-2.0,2.0])
plot_greedy_timestep(dis,pot,10)