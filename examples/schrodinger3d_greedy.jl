using SchrodingerGaussian
using HermiteWavePackets
using Printf
using StaticArrays
using Plots
using LaTeXStrings
using Base.Threads
using JLD2

include("reference_solution.jl")

function plot_schrodinger3d_gaussian(pb::GreedyDiscretization{T}, G0::AbstractWavePacket{D}, V::AbstractWavePacket{D}, Mbox, Nbox; plot_result=true) where{T, D}

    function propag(t, G::AbstractWavePacket{D}) where D
        vt = 2*t * ones(SVector{D, Bool})
        return inv_fourier(unitary_product(fourier(G), Diagonal(vt)))
    end
    function apply_op(t, G)
        return propag(-t, V * propag(t, G))
    end
    @time G_list, res_list = schrodinger_greedy_gaussian(pb, G0, apply_op)

    @time U_ref, X, Y, Z = strange_sine_interaction3d(V, G0, Mbox, Nbox, (pb.tf - pb.t0) / (pb.Lt-1), pb.Lt-1)
    dS = (2*Mbox / (Nbox+1))^3

    if plot_result
        t_list = zeros(pb.Lt)
        norm_list = zeros(pb.Lt)
        diff_list = zeros(pb.Lt)
    
        @threads for k in 1:pb.Lt
            t = pb.t0 + (k-1) * (pb.tf - pb.t0) / (pb.Lt-1)
            G = WavePacketSum{D}(propag.(t, G_list[:, k]))
            
            t_list[k] = t
            norm_list[k] = norm_L2(G)

            diff_list[k] = sqrt(dS) * norm(U_ref[:, :, :, k] - G.(X, Y, Z))
        end

        @save "err_sp80.jld2" t_list diff_list
        plot(t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
        plot!(t_list, fill(norm_L2(G0), pb.Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"), legend=:topright)
        display(plot!())
        # savefig("norm3d.pdf")

        plot(t_list, diff_list;xlabel=LaTeXString("t"), label=LaTeXString("\$L^2\$ norm of the difference"), legend=:bottomright)
        display(plot!())
        # savefig("diff3d.pdf")
    end

    println("res_list = ", res_list)

    # p = plot(; xlabel=LaTeXString("t"))
    # plot!(p, t_list, norm_list; label=LaTeXString("\$ \\Vert ψ(t) \\Vert _{L^2}\$"))
    # plot!(p, t_list, fill(norm_L2(G0), pb.Lt); label=LaTeXString("\$ \\Vert g_0 \\Vert _{L^2}\$"))
    # display(p)
    return diff_list
end

G0 = GaussianWavePacket(complex(1.0), Diagonal(SVector(complex(1.0), complex(1.0), complex(1.0))), SVector(3.0, 3.0, 0.0), SVector(-1/sqrt(2), -1/sqrt(2), 0.0))
G0 = G0 / norm_L2(G0)
pb = GreedyDiscretization(0., 5., 201, 80, 10, true)
V = Gaussian(1.0, Diagonal(SVector(1.0, 1.0, 1.0)), SVector(0.1, 0.0, 0.0))
Mbox = 40.0
Nbox = 80
plot_schrodinger3d_gaussian(pb, G0, V, Mbox, Nbox)